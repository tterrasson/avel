import type { Dispatch } from "../wasm/avel_wasm.js";

export interface DispatchTiming {
  /** Index into `Plan.dispatches`. */
  index: number;
  /** Graph values the dispatch computes, more than one when ops were fused. */
  ops: string[];
  workgroups: [number, number, number];
  /** Nanoseconds since the start of the first dispatch. */
  startNs: number;
  durationNs: number;
}

/** A trace in the Chrome tracing format, for `chrome://tracing` or Perfetto. */
export interface ChromeTrace {
  traceEvents: object[];
  displayTimeUnit: "ns";
}

/** GPU timings of one profiled run, per dispatch. */
export class Profile {
  readonly dispatches: DispatchTiming[];

  /** `timestamps` holds the beginning and end of each dispatch, in nanoseconds. */
  constructor(dispatches: Dispatch[], timestamps: BigInt64Array) {
    const origin = dispatches.length > 0 ? timestamps[0]! : 0n;
    this.dispatches = dispatches.map(({ ops, workgroups }, index) => {
      const start = timestamps[2 * index]!;
      const end = timestamps[2 * index + 1]!;
      return {
        index,
        ops,
        workgroups,
        startNs: Number(start - origin),
        // Some implementations report an end before the start for very short passes.
        durationNs: Math.max(0, Number(end - start)),
      };
    });
  }

  /** Sum of the dispatch durations, without the gaps between them. */
  get totalNs(): number {
    return this.dispatches.reduce((total, { durationNs }) => total + durationNs, 0);
  }

  /** One complete event per dispatch, under a process named `name`. */
  trace(name: string): ChromeTrace {
    const pid = 1;
    const tid = 1;
    return {
      displayTimeUnit: "ns",
      traceEvents: [
        { name: "process_name", ph: "M", pid, args: { name } },
        { name: "thread_name", ph: "M", pid, tid, args: { name: "GPU" } },
        ...this.dispatches.map(({ index, ops, workgroups, startNs, durationNs }) => ({
          name: ops.join(" + "),
          cat: "dispatch",
          ph: "X",
          pid,
          tid,
          ts: startNs / 1000,
          dur: durationNs / 1000,
          args: { index, ops, workgroups },
        })),
      ],
    };
  }
}
