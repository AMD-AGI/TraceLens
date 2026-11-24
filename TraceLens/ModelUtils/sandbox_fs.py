###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The file system interpreted source sees: read-only checkouts plus a scratch store.

Engine and library source reads its own files (tuned-kernel tables, headers,
configs) and sometimes writes derived ones (a merged table under ``/tmp``)
that it reads back. Reads of the checkouts reach the disk; every write lands
in an in-memory store, and later reads of that path see it. Nothing on the
machine changes.

Every route the source can take to a file goes through one
:class:`ScratchFS`: ``open``, ``os`` file descriptors and paths,
``pathlib.Path`` and, for host libraries such as pandas, their own readers and
writers while a run is active.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import stat
import types
from collections.abc import Callable

_EXTENSION_SUFFIXES = (".so", ".pyd", ".dll")


class ScratchFS:
    """Virtual files over a read-only view of the readable directories."""

    def __init__(self, files: dict[str, str], may_read: Callable[[object], bool]):
        self.files = files
        self.may_read = may_read
        self._descriptors: dict[int, tuple[str, io.BytesIO]] = {}
        self._next_fd = 10_000

    # ----------------------------------------------------------- queries

    @staticmethod
    def key(path) -> str:
        return os.path.normpath(os.fspath(path))

    def virtual(self, path) -> str | None:
        try:
            return self.files.get(self.key(path))
        except TypeError:
            return None

    def is_virtual_dir(self, path) -> bool:
        try:
            prefix = self.key(path).rstrip("/") + "/"
        except TypeError:
            return False
        return any(name.startswith(prefix) for name in self.files)

    def built(self, path) -> bool:
        """A compiled extension inside a library checkout: the run's build made it."""
        try:
            text = os.fspath(path)
        except TypeError:
            return False
        return text.endswith(_EXTENSION_SUFFIXES) and self.may_read(
            os.path.dirname(text)
        )

    def exists(self, path) -> bool:
        if (
            self.virtual(path) is not None
            or self.is_virtual_dir(path)
            or self.built(path)
        ):
            return True
        return self.may_read(path) and os.path.exists(path)

    def is_file(self, path) -> bool:
        if self.virtual(path) is not None or self.built(path):
            return True
        return self.may_read(path) and os.path.isfile(path)

    def is_dir(self, path) -> bool:
        if self.is_virtual_dir(path):
            return True
        return self.may_read(path) and os.path.isdir(path)

    def listdir(self, path) -> list[str]:
        names = set()
        if self.may_read(path) and os.path.isdir(path):
            names.update(os.listdir(path))
        prefix = self.key(path).rstrip("/") + "/"
        for name in self.files:
            if name.startswith(prefix):
                names.add(name[len(prefix) :].split("/", 1)[0])
        return sorted(names)

    # -------------------------------------------------------------- files

    def open(self, file, mode="r", *args, **kwargs):
        writing = any(flag in mode for flag in "wax+")
        text = self.virtual(file)
        if writing:
            if "x" in mode and self.exists(file):
                raise FileExistsError(file)
            initial = text if ("a" in mode or "+" in mode) and text is not None else ""
            return self._writer(self.key(file), initial, binary="b" in mode)
        if text is not None:
            if not isinstance(text, str):
                # A checkpoint file stands for weights that were never fetched.
                raise PermissionError(
                    f"{file}: its weights are read through its index only"
                )
            if "b" in mode:
                return io.BytesIO(text.encode("latin-1"))
            return io.StringIO(text)
        if not self.may_read(file):
            raise FileNotFoundError(file)
        return open(file, mode, *args, **kwargs)  # noqa: SIM115 - the caller closes it

    def _writer(self, key, initial, binary):
        store = self.files

        class Writer(io.BytesIO if binary else io.StringIO):
            def close(self):
                value = self.getvalue()
                store[key] = value.decode("latin-1") if binary else value
                super().close()

        writer = Writer(initial.encode("latin-1") if binary else initial)
        writer.seek(0, io.SEEK_END)
        return writer

    def write(self, path, value):
        self.files[self.key(path)] = (
            value.decode("latin-1") if isinstance(value, bytes) else value
        )

    def remove(self, path):
        if self.files.pop(self.key(path), None) is None:
            if self.exists(path):
                raise PermissionError(f"the interpreted source may not remove {path!r}")
            raise FileNotFoundError(path)

    def move(self, source, target):
        text = self.virtual(source)
        if text is None:
            raise PermissionError(f"the interpreted source may not move {source!r}")
        self.files.pop(self.key(source))
        self.files[self.key(target)] = text

    def remove_tree(self, path, ignore_errors=False):
        """``shutil.rmtree``: the scratch files under ``path`` go; real ones stay."""
        prefix = self.key(path).rstrip("/") + "/"
        scratch = [name for name in self.files if name.startswith(prefix)]
        real = self.may_read(path) and os.path.exists(path)
        if real:
            if ignore_errors:
                return
            raise PermissionError(f"the interpreted source may not remove {path!r}")
        if not scratch and not ignore_errors:
            raise FileNotFoundError(path)
        for name in scratch:
            self.files.pop(name)

    def move_tree(self, source, target):
        """``shutil.move``: a scratch file or directory changes its name."""
        if self.virtual(source) is not None:
            self.move(source, target)
            return target
        prefix = self.key(source).rstrip("/") + "/"
        scratch = [name for name in self.files if name.startswith(prefix)]
        if not scratch:
            raise PermissionError(f"the interpreted source may not move {source!r}")
        destination = self.key(target).rstrip("/") + "/"
        for name in scratch:
            self.files[destination + name[len(prefix) :]] = self.files.pop(name)
        return target

    def stat(self, path):
        text = self.virtual(path)
        if text is not None:
            size = len(text) if isinstance(text, str) else text.size
            return os.stat_result((stat.S_IFREG | 0o644, 0, 0, 1, 0, 0, size, 0, 0, 0))
        if self.is_virtual_dir(path):
            return os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        if not self.may_read(path):
            raise FileNotFoundError(path)
        return os.stat(path)

    # ---------------------------------------------------- file descriptors

    def os_open(self, path, flags, mode=0o777, *args, **kwargs):
        writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)
        if not writes:
            text = self.virtual(path)
            if text is None:
                if not self.may_read(path):
                    raise FileNotFoundError(path)
                return os.open(path, flags)
            buffer = io.BytesIO(text.encode("latin-1"))
        else:
            if flags & os.O_EXCL and self.exists(path):
                raise FileExistsError(path)
            existing = self.virtual(path) if not flags & os.O_TRUNC else None
            buffer = io.BytesIO((existing or "").encode("latin-1"))
            buffer.seek(0, io.SEEK_END)
            self.files[self.key(path)] = existing or ""
        descriptor = self._next_fd
        self._next_fd += 1
        self._descriptors[descriptor] = (self.key(path), buffer)
        return descriptor

    def os_write(self, descriptor, data):
        if descriptor not in self._descriptors:
            raise PermissionError("the interpreted source may only write scratch files")
        key, buffer = self._descriptors[descriptor]
        written = buffer.write(bytes(data))
        self.files[key] = buffer.getvalue().decode("latin-1")
        return written

    def os_read(self, descriptor, size):
        if descriptor not in self._descriptors:
            return os.read(descriptor, size)
        return self._descriptors[descriptor][1].read(size)

    def os_close(self, descriptor):
        if self._descriptors.pop(descriptor, None) is None:
            os.close(descriptor)

    # ------------------------------------------------------------ modules

    def install_os(self, module: types.ModuleType) -> None:
        """Route ``module`` (the source's ``os``) through this store."""
        module.open = self.os_open
        module.write = self.os_write
        module.read = self.os_read
        module.close = self.os_close
        module.fsync = lambda descriptor: None
        module.fdopen = lambda descriptor, mode="r", *args, **kwargs: _Descriptor(
            self, descriptor, mode
        )
        module.remove = module.unlink = self.remove
        module.replace = module.rename = self.move
        module.stat = module.lstat = lambda path, *args, **kwargs: self.stat(path)
        module.listdir = lambda path=".": self.listdir(path)
        module.makedirs = lambda *args, **kwargs: None
        module.mkdir = lambda *args, **kwargs: None
        module.rmdir = lambda *args, **kwargs: None
        module.utime = lambda *args, **kwargs: None
        module.access = lambda path, mode, *args, **kwargs: self.exists(path)
        module.path.exists = module.path.lexists = self.exists
        module.path.isfile = self.is_file
        module.path.isdir = self.is_dir
        module.path.getsize = lambda path: self.stat(path).st_size
        module.path.getmtime = lambda path: self.stat(path).st_mtime

    def pathlib_module(self) -> types.ModuleType:
        """``pathlib`` whose concrete paths read and write through this store."""
        fs = self
        module = types.ModuleType("pathlib")
        module.__dict__.update(
            {
                key: value
                for key, value in vars(pathlib).items()
                if not key.startswith("__")
            }
        )

        class Path(pathlib.PosixPath):
            def exists(self, *args, **kwargs):
                return fs.exists(self)

            def is_file(self, *args, **kwargs):
                return fs.is_file(self)

            def is_dir(self, *args, **kwargs):
                return fs.is_dir(self)

            def stat(self, *args, **kwargs):
                return fs.stat(self)

            def open(self, mode="r", *args, **kwargs):
                return fs.open(self, mode, *args, **kwargs)

            def read_text(self, *args, **kwargs):
                with fs.open(self, "r") as handle:
                    return handle.read()

            def read_bytes(self):
                with fs.open(self, "rb") as handle:
                    return handle.read()

            def write_text(self, data, *args, **kwargs):
                fs.write(self, data)
                return len(data)

            def write_bytes(self, data):
                fs.write(self, data)
                return len(data)

            def touch(self, *args, **kwargs):
                if fs.virtual(self) is None:
                    fs.write(self, "")

            def mkdir(self, *args, **kwargs):
                pass

            def unlink(self, missing_ok=False):
                try:
                    fs.remove(self)
                except FileNotFoundError:
                    if not missing_ok:
                        raise

            def rename(self, target):
                fs.move(self, target)
                return type(self)(target)

            replace = rename

            def iterdir(self):
                for name in fs.listdir(self):
                    yield self / name

            def glob(self, pattern, *args, **kwargs):
                import fnmatch

                found = set()
                if fs.may_read(self):
                    found.update(
                        str(path) for path in pathlib.PosixPath(self).glob(pattern)
                    )
                found.update(
                    name
                    for name in fs.files
                    if fnmatch.fnmatch(name, os.path.join(fs.key(self), pattern))
                )
                return iter(sorted(type(self)(name) for name in found))

            def rglob(self, pattern, *args, **kwargs):
                return self.glob(f"**/{pattern}")

        module.Path = module.PosixPath = Path
        return module

    @contextlib.contextmanager
    def host_libraries(self):
        """Point host libraries' own file readers and writers at this store.

        pandas reads and writes tables by path from its own code; while a run
        is active, a path in the store reads from it and a write lands in it.
        """
        try:
            import pandas
        except ImportError:
            yield
            return
        read_csv = pandas.read_csv
        to_csv = pandas.DataFrame.to_csv
        fs = self

        def scratch_read_csv(path, *args, **kwargs):
            if isinstance(path, str | os.PathLike):
                text = fs.virtual(path)
                if text is not None:
                    return read_csv(io.StringIO(text), *args, **kwargs)
                if not fs.may_read(path):
                    raise FileNotFoundError(path)
            return read_csv(path, *args, **kwargs)

        def scratch_to_csv(frame, path_or_buf=None, *args, **kwargs):
            if isinstance(path_or_buf, str | os.PathLike):
                fs.write(path_or_buf, to_csv(frame, None, *args, **kwargs))
                return None
            return to_csv(frame, path_or_buf, *args, **kwargs)

        pandas.read_csv = scratch_read_csv
        pandas.DataFrame.to_csv = scratch_to_csv
        try:
            yield
        finally:
            pandas.read_csv = read_csv
            pandas.DataFrame.to_csv = to_csv


class _Descriptor(io.RawIOBase):
    """``os.fdopen`` of a scratch descriptor."""

    def __init__(self, fs: ScratchFS, descriptor: int, mode: str):
        self._fs = fs
        self._descriptor = descriptor
        self._text = "b" not in mode

    def write(self, data):
        payload = data.encode() if isinstance(data, str) else data
        return self._fs.os_write(self._descriptor, payload)

    def read(self, size=-1):
        data = self._fs.os_read(self._descriptor, size)
        return data.decode() if self._text else data

    def close(self):
        if not self.closed:
            self._fs.os_close(self._descriptor)
        super().close()

    def writable(self):
        return True

    def readable(self):
        return True
