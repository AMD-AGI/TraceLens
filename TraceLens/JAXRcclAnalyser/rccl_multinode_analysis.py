import argparse
import sys
from glob import glob
from pathlib import Path
import re
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
import os

def dict_add(dict,key,value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key]=[value]
    return dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process JAX trace communications"
    )
    parser.add_argument(
        "--xla-dump",
        "-x",
        type=str,
        help="Input xla module dump or directory containing the XLA dump of the traces. If this is a directory then the last module found will be used",
    )

    parser.add_argument(
        "--xplane-dir",
        "-t",
        type=str,
        help="Folder path containing the profiler xplane.pb file",
    )

    parser.add_argument(
        "--module-name",
        type=str,
        help="Name of the module, example: train_step",
    )

    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        help="Input profile or directory a Jax profile. If this is a directory the the script will travers single directories until the leaf and attempt to find it",
    )

    parser.add_argument(
        "--device", "-d", type=str, default="mi300x", help="Device type (default 'mi300x')"
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        help="Resolution of the model (e.g., 544x960x9)",
        default=None,
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        help="Additional metadata to include in filename",
        default=None,
    )
    parser.add_argument(
        "--plot-dir",
        "-pd",
        type=str,
        help="Output directory for plots",
        default="plots",
    )

    parser.add_argument(
        "--gpus",
        "-g",
        type=int,
        help="Number of gpus used to generate the traces",
        default=8,
    )
    return parser.parse_args()

# Find the matching file of the largest size, not clear why ?
# if this is required probably there is a better way to write this function
# This might not get the required files from all the participating nodes if the
# logs structure follow how we log from the maxtext runs
def detect_xla_file(dir: str) -> str:  
    filenames=glob(f"{dir}/**/*_gpu_after_optimizations-buffer-assignment.txt", recursive = True)
    selected=filenames[0]
    max_size=os.path.getsize(filenames[0])
    for filename in filenames:
        size = os.path.getsize(filename)
        if size > max_size:
            max_size=size
            selected=filename
    return selected

def detect_profile(dir: str) -> str :
    return next(Path(dir).rglob("*.trace.json.gz"))

# What is this events map ?
events_map={"all-gather-start":"all_gather", "all-reduce-start":"all_reduce", "reduce-scatter":"reduce_scatter", "collective-permute-start": "collective_permute","all-to-all":"alltoall"}
#events_map={ "reduce-scatter":"reduce_scatter"}

def read_xla_dump(file: str) -> dict:
    return_value={}
    replica_groups={}
    for event_key,event_val in events_map.items(): # Nested loops should be switched (outer -> inner)
        print(event_key)
        for line in open(file, "r"): # Resource handling is not proper
            if True: 
                re_pattern = re.compile(f"value:.*({event_key}[.0-9]*)"+"[{ ].*size=([0-9]*),.*( [sbfp3214568nuzem]*)\[")
                collective_name=(re_pattern.search(line))
                end_paren=re.search(".*\)$",line)
                if collective_name and end_paren is None:
                    if collective_name.group(1) in return_value:
                        return_value[collective_name.group(1)].append([collective_name.group(2),collective_name.group(3).strip(" ")])
                    else:
                        return_value[collective_name.group(1)]=[[collective_name.group(2),collective_name.group(3).strip(" ")]]
                    continue
                if "replica_groups" in line:
                    #print(line)
                    re_pattern = re.compile(f"from instruction: %({event_key}[.0-9]*)"+"[{ ].*replica_groups=([0-9,\[\]<=\{\}]*)")
                    collective_name=(re_pattern.search(line))
                    if collective_name is not None:
                        replica_groups=dict_add(replica_groups,collective_name.group(1),collective_name.group(2))
    #print(return_value)
    return return_value, replica_groups

def parse_rccl_output(filename):
    found=0
    line_number=0
    with open(filename, 'r') as file:
        for line in file:
            line_number+=1
            if "#       size" in line:
                found=line_number+2
            if line_number==found:
                data=(re.sub(' +', ' ', line).strip(" ").split(" "))
                return data[5],data[6],data[7]

    return 0,0,0
            # Process the line (e.g., print it, store it, etc.)
            #print(line.strip())  # strip() removes leading/trailing whitespace, including newline characters
def convert_size(size):
    if size<100:
        return f'{(size):.2f}B'
    if size<100000:
        return f'{(size/10**3):.2f}KB'
    if size<100000000:
        return f'{(size/10**6):.2f}MB'
    return f'{(size/10**9):.2f}GB'
def merge_traces(traceevents: list) -> list:
    num_nodes=len(traceevents)
    merged_traceevents=[]
    for i in range(0,num_nodes):
        te=traceevents[i]
        print(type(te))
        print(len(te))
        offset=int(i*8)
        pids=[]
        for event in te:
            #print(i)
            if "pid" in event.keys() and "dur" in event.keys():
                if not(event["pid"] == 701):
                    event["pid"]=event["pid"]+offset
                    merged_traceevents.append(event)
                    pids.append(event["pid"])

    return merged_traceevents

def combine_xla_trace(coll_dict: dict,rccl_dict: dict, wait_rccl_dict: dict,num_nodes=1,xplane_dir="./") -> dict:
    #reduce-scatter: the count is nranks \times the buffer size. For all-gather, all-reduce, all-to-all, and collective permute, the message size is the same as the call
    size_dict={"bf16":2,"f32":4,"s32":4,"fp16":2,"f64":8,"int32":4,"int16":2,"f8e4m3fnuz":1,"f8e5m2fnuz":1}
    name_dict={"bf16":"bfloat16","f32":"float","s32":"int32","f8e4m3fnuz":"fp8_e4m3","f8e5m2fnuz":"fp8_e5m2"}
    command_dict={"all-gather":"all_gather_perf","all-reduce":"all_reduce_perf","reduce-scatter":"reduce_scatter_perf","all-to-all":"alltoall_perf"}
    full_dict={}
    num_gpus=num_nodes*8
    host_ips=["tw028","tw044","tw046","tw049"]
    rccl_test_dict={"all-gather-start":(1316.3,133.83,0),"all-gather-start.1":(11167,54.09,0),"all-gather-start.2":(1220.6,144.32,0),"all-gather-start.3":(10604,56.96,0),"all-gather-start.4":(2426.2,331.93,0),"all-reduce-start":(72.68,0.34,0),"all-reduce-start.1":(58.5,0,0),"all-reduce-start.2":(60.28,0,0),"all-reduce-start.3":(73.04,0.17,0),"all-reduce-start.4":(59.97,0,0),"all-reduce-start.5":(59.54,0,0),"reduce-scatter.13":(9474.4,63.75,0),"reduce-scatter.14":(1379.9,127.66,0),"reduce-scatter.15":(2772.2,290.49,0),"all-to-all.2.1":(6571,337.02,0),"all-to-all.3.1":(6566,337.28,0),"all-to-all.4.1":(6570,337.08,0),"all-to-all.7.1":(6565.2,337.32,0),"all-to-all.8.1":(6573.3,336.91,0)}
    print_summary="collective,size,dtype,#calls,mean_latency(us),rccl_test_latency(us),mean_algBW,rccl_test_algBW\n"
    for k,v in coll_dict.items():
        if k not in rccl_dict.keys():
            print(f"{k} event not found in the trace")
            continue
        hosts=""
        hosts_mt=""
        full_dict[k]={}
        host_list=[0 for i in range(0,num_nodes)]
        involved_hosts=0
        if ("<=" in v[0][0]) and len(v[0])==1:
            num_groups=1
            group_size=num_gpus
            for nid in range(0,num_nodes):
                hosts+=(host_ips[nid]+":8,")
                hosts_mt+=(host_ips[nid]+":1,")
            involved_hosts=num_nodes
        else:
            num_groups=len(v[0])
            group_size=len(v[0][0].split(","))
            groups=v[0][0].split(",")
            for gid in groups:
                g=gid.strip("{").strip("}")
                host_list[(int(g))//8]+=1
        
        for nid in range(0,num_nodes):
            if host_list[nid]>0:
                involved_hosts+=1
                hosts+=(host_ips[nid]+":"+str(host_list[nid])+",")
                hosts_mt+=(host_ips[nid]+":1,")
        assert int(num_groups*group_size)==num_gpus,"total number of gpus in replica group is not 8"
        #scale_count=size_dict.get(v[2],1)
        message_size=v[1]  # buffer size
        if "reduce-scatter" in k:
            message_size=message_size*group_size
        #print(k,rccl_dict[k])
        eff_bw=[message_size/(i*1000) for i in rccl_dict[k]]

        # AM : Commented out the code for RCCL-tests
        
        # commandname=[name  for reference,name in command_dict.items() if reference in k][0]
        # #message=f'mpirun -np {group_size} --hosts {hosts.strip(",")} /home/devashah@amd.com/rccl_test_2/rccl-tests/build/{commandname} -b {message_size} -e {message_size} -f 2 -g 1 -c 0 -d {name_dict.get(v[2],"float")}'
        
        # message=f'/usr/local/bin/ompi/bin/mpirun -np {group_size} -H {hosts.strip(",")}   --mca pml ucx   --mca btl ^openib   -x NCCL_SOCKET_IFNAME=ens51np0   -x NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1   -x NCCL_IB_GID_INDEX=3   -x NCCL_MIN_NCHANNELS=112   -x NCCL_DEBUG=version   /usr/local/bin/rccl-tests/build/{commandname} -b {message_size} -e {message_size} -f 2 -g 1 -d {name_dict.get(v[2],"float")}'
        # #message=f'/usr/local/bin/ompi/bin/mpirun -np {involved_hosts} -H {hosts_mt.strip(",")}   --mca pml ucx   --mca btl ^openib   -x NCCL_SOCKET_IFNAME=ens51np0   -x NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1   -x NCCL_IB_GID_INDEX=3   -x NCCL_MIN_NCHANNELS=48   -x NCCL_DEBUG=version   /usr/local/bin/rccl-tests/build/{commandname} -b {message_size} -e {message_size} -f 2 -g 1 -t {int(group_size/involved_hosts)} -G 2 -d {name_dict.get(v[2],"float")}'
        
        # print(f'running {message}')
        # os.system(f'{message} | tee temp')
        #os.system(f'bash /home/devashah@amd.com/rccl_test_2/run_command.sh "{message}" | tee temp')        
        # latency,algbw,busbw=parse_rccl_output("temp")

        latency, algbw, busbw = 0, 0 ,0 
        print_summary+=str(f'{k},{message_size},{v[2]},{len(rccl_dict[k])},{np.mean(rccl_dict[k]):.0f},{latency},{message_size/(1000*np.mean(rccl_dict[k])):.0f},{algbw}\n')
        
        #latency,algbw,busbw=rccl_test_dict[k]
        # number of gourps, group_size, data type, message size in B, number of calls, effective bandwidth, runtime latency, rccl test latency, rccl test algBW 
        full_dict[k]=[num_groups,group_size,name_dict[v[2]],message_size,len(rccl_dict[k]),eff_bw,rccl_dict[k],wait_rccl_dict[k],latency,algbw,busbw]
    print("######")
    print(print_summary)
    print("######")
    for collective in ["all-gather","all-reduce","all-to-all","reduce-scatter"]:
        data = []
        bw=[]
        labels=[]
        sizes=[]
        num_messages=[]
        gemmdf = pd.DataFrame(columns=['size','bw'])
        count=0
        pure_total_runtime=0
        total_runtime=0
        projected_runtime=0
        for k,v in full_dict.items():
            if collective in k:
                pure_total_runtime+=np.sum(v[6])
                total_runtime+=np.sum(v[7])
                total_runtime+=np.sum(v[6])
                num_messages.append(len(v[6]))
                projected_runtime+=len(v[6])*float(v[8])
                projected_runtime+=np.sum(v[7])
                print(len(v[6]))
                data.append(v[5])
                bw.append(float(v[9]))
                labels.append(convert_size(v[3])+"\nGroupSize-"+str(v[1]) + "\n #Calls-"+str(int(len(v[6])/8)))
                sizes.append(int(v[3]))
        print("Projection",collective,pure_total_runtime,total_runtime,projected_runtime)
        # fig, ax = plt.subplots()
        # data=[x for _,x in sorted(zip(sizes,data))]
        # labels=[x for _,x in sorted(zip(sizes,labels))]
        # bw=[x for _,x in sorted(zip(sizes,bw))]

        # # Scatter plot (overlayed)
        
        
        # ax.boxplot(data,label="Runtime algBW")
        # ax.scatter(y=bw,x=range(1,len(bw)+1),marker="*",color="red",label="Standalone rccl-test algBW",s=100)
        # #print(labels)
        # ax.set_xticklabels(labels,rotation=45,fontsize=9)
        # #ax.set_yticks([0,100,200,300,400])
        # #ax.set_yticklabels(["0","100","200","300","400"])
        # #ax.set_ylim(ymin=0,ymax=400)
        # plt.legend()
        # plt.title(f'Collective {collective} Runtime: {total_runtime/32000:.0f}ms, Projected: {projected_runtime/32000:.0f}ms')
        # #plt.xlabel('Message size,')
        # plt.ylabel('BW (GB/sec)')
        # plt.savefig(f'{xplane_dir}/rccl_roofline_{collective}.pdf', bbox_inches="tight", dpi=300)
        
        
def combine_xla_dump(coll_dict: dict,replica_groups: dict) -> dict:
    coll_dict_compressed={}
    
    for k,v in replica_groups.items():
        if len(list(set(v)))>1:
            print(f'different replica groups found for {k}')
        coll_dict_compressed[k]=[v[0].strip(",").split(",{")]

    for k,v in coll_dict.items():
        # print(k)
        message=0
        dtype=v[0][1]
        for i in v:
            message+=int(i[0]) # Even if the dtype of the messgaes are not same still message size summation is happening
            if i[1] != dtype:
                print("datatype in combined messages not working?")      
        if k in coll_dict_compressed:
            # print(k)
            coll_dict_compressed[k].append(message)
            coll_dict_compressed[k].append(dtype)

    return coll_dict_compressed

def read_profile(traceevents: list, num_nodes=1) -> dict:
    def get_unique_sizes(mdict, message):
        uniq={}
        for i in mdict.keys():
            if message in i:
                size=mdict[i][1]
                uniq[size] = uniq.get(size, 0) + 1
        return uniq
    num_gpus=8*num_nodes
    collective_events=[[] for i in range(0,num_gpus)]
    merged_collective_events=[[] for i in range(0,num_gpus)]
    rccl={}
    wait_rccl={}
    for i in traceevents:
        print(i)
        if "pid" in i.keys() and "dur" in i.keys():
            if not(i["pid"] == 701):
                name=i["name"]
                dur=i["dur"]
                pid=i["pid"]
                tid=i["tid"]
               
                #TODo: figure out the msccl thing
                if "rccl" in name or "nccl" in name:
                    op = i["args"]["hlo_op"]
                    collective_events[pid-1].append([op,name,dur,i["ts"],i["ts"]+dur])
                    # if "reduce-scatter.10.1" in op:
                    #     print(i)
                    #if op.startswith('reduce-scatt
                    # er'):
                    #    op = '.'.join(op.split('.')[:2]) # need to remove sub-communications from reduce-scatter only

    for i in range(num_gpus):
        collective_events[i].sort(key=lambda x: x[3])
    #print((collective_events[0][-1]),(collective_events[5][-1]))
    
    #sys.exit()  
    for i in range(num_gpus):
        current=collective_events[i][0]
        k=0
        while k<len(collective_events[i])-1:
            current=collective_events[i][k]
            next=collective_events[i][k+1]
            if current[0]==next[0] and not (current[1]==next[1]):
                #print(f'merging {current[1]} and {next[1]} kernels')
                #print(current,next)
                merged_collective_events[i].append([current[0],next[4]-current[3],current[3],next[4]])
                k+=1
            else:
                #hlo_op_name, duration, start time, end time
                merged_collective_events[i].append([current[0],current[2],current[3],current[4]])
            k+=1
        current=collective_events[i][k]
        merged_collective_events[i].append([current[0],current[2],current[3],current[4]])
    #sys.exit()
    #for i in range(0,num_gpus):
    #    print(len(merged_collective_events))
    #for i in merged_collective_events[5]:
    #   print(i)   
    #sys.exit()
    # queue_len=len(merged_collective_events[0])
    queue_len = 5926
    print(f"Queue len : {queue_len}")
                   
    for j in range(0,queue_len): 
        runtime= merged_collective_events[0][j][1]
        name=merged_collective_events[0][j][0]
        for i in range(0,num_gpus):
            c_runtime= merged_collective_events[i][j][1]
            c_name=merged_collective_events[i][j][0]
            if c_name==name:
                if c_runtime < runtime:
                    runtime=c_runtime
            else:
                print("name mismatch",c_name,name)
        #assert len(merged_collective_events[i])==queue_len,"the rccl event numbers do not match across ranks"
        #for j in range(0,queue_len):
        #for j in range(0,len(merged_collective_events[i])):
        for i in range(0,num_gpus):
            c_runtime= merged_collective_events[i][j][1]
            c_name=merged_collective_events[i][j][0]
            if c_name==name:
                rccl=dict_add(rccl,name,runtime)
                wait_rccl=dict_add(wait_rccl,name,c_runtime-runtime)
            else:
                print("name mismatch",c_name,name)
    #print(rccl["all-gather-start.1"])
    #print(rccl["all-gather-start"])
    #sys.exit()
    return rccl, wait_rccl

    #each dict is indexed by the hlo_op, and the value is a list [duration, total message size, number of tuple arguments,algbw]
    output = {}
    for msg_type, msg_values in messages.items():
        coll_dict={}
        output[events_map[msg_type]] = coll_dict
        for msg in msg_values:
            collname=f"{msg_type}.{msg[0]}" if msg[0] is not None else msg_type
            collsize=int(msg[1])
            collval = rccl.get(collname, None)
            if (collval is not None):
                current = coll_dict.get(collname, [min(collval),0,0,0])
                current[1] += collsize
                current[2] += 1
                coll_dict[collname] = current
            else:
                print(collname," not found")
        scale = gpus if "reduce-scatter" in msg_type else 1
        for collname, current in coll_dict.items():
            current[3]=current[1]*scale*0.001/current[0]

    return output

def generate_plots(plot_dir: str, profile_data: dict, device: str, resolution=None, metadata=None) -> None:
    def generate_output_filename(collective, device, resolution=None, metadata=None):
        parts = [collective, device]
        if resolution:
            parts.append(resolution)
        if metadata:
            parts.append(metadata)
        return "_".join(parts) + "_stats.png"
    for collective, collective_stats in profile_data.items():
        current_data = [[collective, xfer_name, data[0], data[1] / 1024, data[3]]
                        for xfer_name, data in collective_stats.items()]
        df = pd.DataFrame(data=current_data,
                          columns = [
                              "base_collective",
                              "collective_name",
                              "latency_us",
                              "buffer_size_kb",
                              "effective_bw" ])

        # Group by collective type and buffer size for bandwidth stats
        bandwidth_stats = (
            df.groupby(["base_collective", "buffer_size_kb"])["effective_bw"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Group by collective type and buffer size for call counts
        call_counts = (
            df.groupby(["base_collective", "buffer_size_kb"])
            .size()
            .reset_index(name="count")
        )

        # Define a color palette for better visualization
        colors = [
            "#FF9999",
            "#66B2FF",
            "#99FF99",
            "#FFCC99",
            "#FF99CC",
            "#99FFCC",
            "#FFB366",
            "#FF99FF",
        ]

        # Define bandwidth thresholds in GB/s - adjusted for better spread
        bw_thresholds = [0, 50, 100, 200, 300, 350]

        # Create figure with three subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))

        # Get data for this collective
        collective_data = df[df["base_collective"] == collective]
        bw_data = bandwidth_stats[
            bandwidth_stats["base_collective"] == collective
        ].sort_values("buffer_size_kb")
        count_data = call_counts[
            call_counts["base_collective"] == collective
        ].sort_values("buffer_size_kb")

        # Plot 1: Bandwidth
        ax1.bar(
            range(len(bw_data)),
            bw_data["mean"],
            yerr=bw_data["std"],
            capsize=5,
            color="skyblue",
            width=0.6,
        )

        ax1.set_xlabel("Buffer Size (KB)")
        ax1.set_ylabel("Bandwidth (GB/s)")
        title = f"{device.capitalize()}: {collective} Bandwidth vs Buffer Size"
        if resolution:
            title += f" ({resolution})"
        ax1.set_title(title)
        ax1.set_xticks(range(len(bw_data)))
        ax1.set_xticklabels(
            [f"{x:.0f}" for x in bw_data["buffer_size_kb"]], rotation=45
        )
        ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

        # Add bandwidth values on top of bars
        for i, v in enumerate(bw_data["mean"]):
            std_ = 0 if np.isnan(bw_data["std"].iloc[i]) else bw_data["std"].iloc[i]
            ax1.text(i, v + std_, f"{v:.1f}", ha="center", va="bottom")

        # Plot 2: Call Counts
        ax2.bar(
            range(len(count_data)),
            count_data["count"],
            color="lightgreen",
            width=0.6,
        )

        ax2.set_xlabel("Buffer Size (KB)")
        ax2.set_ylabel("Number of Calls")
        title = f"{device.capitalize()}: {collective} Call Count vs Buffer Size"
        if resolution:
            title += f" ({resolution})"
        ax2.set_title(title)
        ax2.set_xticks(range(len(count_data)))
        ax2.set_xticklabels(
            [f"{x:.0f}" for x in count_data["buffer_size_kb"]], rotation=45
        )
        ax2.grid(True, axis="y", linestyle="--", alpha=0.7)

        # Add count values on top of bars
        for i, v in enumerate(count_data["count"]):
            ax2.text(i, v, str(v), ha="center", va="bottom")

        # Plot 3: Time distribution across buffer sizes
        time_by_size = (
            collective_data.groupby("buffer_size_kb")["latency_us"].sum().reset_index()
        )
        total_time_us = time_by_size["latency_us"].sum()
        time_by_size["percentage"] = (time_by_size["latency_us"] / total_time_us) * 100

        ax3.bar(
            range(len(time_by_size)),
            time_by_size["percentage"],
            color="orchid",
            width=0.6,
        )

        ax3.set_xlabel("Buffer Size (KB)")
        ax3.set_ylabel("% of Total Time")
        title = f"{device.capitalize()}: {collective} Time Distribution by Buffer Size"
        if resolution:
            title += f" ({resolution})"
        ax3.set_title(title)
        ax3.set_xticks(range(len(time_by_size)))
        ax3.set_xticklabels(
            [f"{x:.0f}" for x in time_by_size["buffer_size_kb"]], rotation=45
        )
        ax3.grid(True, axis="y", linestyle="--", alpha=0.7)

        # Add percentage values on top of bars
        for i, v in enumerate(time_by_size["percentage"]):
            if v > 0.5:  # Only show if >0.5%
                ax3.text(
                    i,
                    v,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        # Plot 4: Time spent in different bandwidth ranges
        total_time = (collective_data["latency_us"].sum()) / 1e6  # Convert to seconds

        # Calculate time spent in each bandwidth range
        time_in_ranges = []
        labels = []

        for i in range(len(bw_thresholds) - 1):
            mask = (collective_data["effective_bw"] >= bw_thresholds[i]) & (
                collective_data["effective_bw"] < bw_thresholds[i + 1]
            )
            time_in_range = (collective_data[mask]["latency_us"].sum()) / 1e6
            percentage = (time_in_range / total_time) * 100
            time_in_ranges.append(percentage)
            labels.append(f"{bw_thresholds[i]}-{bw_thresholds[i+1]} GB/s")

        # Add the highest range
        mask = collective_data["effective_bw"] >= bw_thresholds[-1]
        time_in_range = (collective_data[mask]["latency_us"].sum()) / 1e6
        percentage = (time_in_range / total_time) * 100
        time_in_ranges.append(percentage)
        labels.append(f"≥{bw_thresholds[-1]} GB/s")

        # Create stacked bar chart for time distribution
        for i, (percentage, label, color) in enumerate(
            zip(time_in_ranges, labels, colors)
        ):
            bottom = sum(time_in_ranges[:i])
            ax4.bar(
                0,
                percentage,
                bottom=bottom,
                label=label,
                color=color,
                width=0.5,
                edgecolor="black",
                linewidth=1,
            )  # Add black edges

        # Adjust the appearance
        ax4.set_ylabel("Percentage of Total Time (%)", fontsize=10)
        title = (
            f"{device.capitalize()}: {collective} Time Distribution by Bandwidth Range"
        )
        if resolution:
            title += f" ({resolution})"
        title += f"\nTotal Time: {total_time:.3f}s"
        ax4.set_title(title, fontsize=12, pad=20)

        # Move legend outside and make it more readable
        ax4.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
            frameon=True,
            edgecolor="black",
        )

        # Add percentage labels with better visibility
        total = 0
        for percentage in time_in_ranges:
            if percentage > 1:  # Show labels for segments >1%
                ax4.text(
                    0,
                    total + percentage / 2,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
                )
            total += percentage

        # Adjust layout to prevent text overlap
        plt.tight_layout()
        # Save file
        output_filename = generate_output_filename(
            collective, device, resolution, metadata
        )
        output_path = Path(plot_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / output_filename, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Generated: {output_filename}")



if __name__ == "__main__":

    # args = parse_args()
    # xla_dump = args.xla_dump
    xla_dump_dir = "/workspace/code/JaxTrace_Analysis/logs/20250509_1604_mixtral/tw026/xla_dumps"

    if Path(xla_dump_dir).is_dir():
        print(f"Detecting XLA module from directory {xla_dump_dir}", file=sys.stderr)
        xla_dump = detect_xla_file(xla_dump_dir)
        print(f"Selected XLA module {xla_dump}", file=sys.stderr)
    if not Path(xla_dump).is_file():
        print(f"XLA module {xla_dump} does not exist", file=sys.stderr)
        sys.exit(1)

    messages,replica_groups = read_xla_dump(xla_dump)
    
    messages_combine= combine_xla_dump(messages,replica_groups)
    for k,v in messages_combine.items():
        print(k,v)

    if len(sys.argv) < 2:
        print("Usage: python3 collective_analysis.py <folder_path_with_xplane> <module_name>")
        sys.exit(1)

    logs_dir = "/workspace/code/JaxTrace_Analysis/logs/20250509_1604_mixtral"
    xplane_files=glob(f"{logs_dir}/**/*.xplane.pb", recursive = True)
    gzip_files=glob(f"{logs_dir}/**/*.trace.json.gz", recursive = True)


    print(xplane_files)
    print(f'Multinode analysis with {len(xplane_files)} nodes')

    # Get the trace events dictionary from the xplane.pb file
    num_nodes=len(xplane_files)
    traceevents=[[] for i in range(0,num_nodes)]
    
    from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
    for i in range(0,num_nodes):
        #traceevents[i]=xla_parser.get_traceevents(xplane_files[i])
        import gzip
        with gzip.open(gzip_files[i], 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
        #result, _ = convert.xspace_to_tool_data([xplane_files[i]], "trace_viewer@^", {})
        #result = result.decode("utf-8") # we get bytes back from the call above
        #with open(xplane_files[i].replace("pb", "processed.json"), 'w') as writefile:
        #    writefile.write(result)
        #data=json.loads(result)
        traceevents[i]=data["traceEvents"]
    traceevents=merge_traces(traceevents)
    #traceevents=xla_parser.get_traceevents(xplane_files[0])
    # Get the hlo protob pb file line data from the xplane.pb file
    ##hlo_pb_line_data=xla_parser.get_hlo_pb(xplane_dir,xplane_files,module_name)

    rccl_dict, wait_rccl_dict=read_profile(traceevents,num_nodes=num_nodes)
    #sys.exit()
    combine_xla_trace(messages_combine,rccl_dict, wait_rccl_dict,num_nodes=num_nodes,xplane_dir="")
    #print(messages_combine)
    # sys.exit()
    # profile = args.profile
    # if Path(profile).is_dir():
    #     print(f"Detecting profile from directory {profile}", file=sys.stderr)
    #     profile = detect_profile(profile)
    #     print(f"Selected profile {profile}", file=sys.stderr)
    # if not Path(profile).is_file():
    #     print(f"Profile {profile} does not exist", file=sys.stderr)
    #     sys.exit(1)

    

    # profile_data = read_profile(profile, args.gpus, messages)

    # generate_plots(args.plot_dir, profile_data, args.device, args.resolution, args.metadata)