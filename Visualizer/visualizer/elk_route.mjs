#!/usr/bin/env node
/** Read an ELK JSON graph on stdin, layout with elkjs, write result JSON to stdout. */
import ELK from "elkjs/lib/elk.bundled.js";
import { readFileSync } from "node:fs";

const elk = new ELK();
const input = JSON.parse(readFileSync(0, "utf8"));

try {
  const result = await elk.layout(input);
  process.stdout.write(JSON.stringify(result));
} catch (error) {
  console.error(String(error?.stack || error));
  process.exit(1);
}
