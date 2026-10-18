import assert from "node:assert/strict";
import { test } from "node:test";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/packages/engine-lib/src/PseudoRandom";
import { LatencyQueue } from "./latency";

const intent = (tile: number): StampedIntent => ({ type: "spawn", tile, clientID: "AGENT000" });

test("delays are 1, 2 or 3 turns with p = .6/.3/.1", () => {
  const q = new LatencyQueue(new PseudoRandom(42));
  const counts = [0, 0, 0];
  for (let i = 0; i < 10000; i++) {
    q.push(intent(i), 100);
    for (let d = 0; d < 3; d++) if (q.take(100 + d).length > 0) counts[d]++;
  }
  assert.deepEqual(counts.map((c) => Math.round(c / 1000)), [6, 3, 1]);
});

test("same seed, same delays", () => {
  const run = () => {
    const q = new LatencyQueue(new PseudoRandom(7));
    const due: number[] = [];
    for (let i = 0; i < 50; i++) {
      q.push(intent(i), 0);
      for (let t = 0; t < 3; t++) if (q.take(t).length > 0) due.push(t);
    }
    return due;
  };
  assert.deepEqual(run(), run());
});

test("intents due in the same turn come out in varying order", () => {
  const q = new LatencyQueue(new PseudoRandom(3));
  const firsts = new Set<number>();
  for (let i = 0; i < 200; i++) {
    const turn = i * 10;
    q.push(intent(1), turn);
    q.push(intent(2), turn);
    const out = [...q.take(turn), ...q.take(turn + 1), ...q.take(turn + 2)];
    const first = out[0];
    if (out.length === 2 && first.type === "spawn") firsts.add(first.tile);
  }
  assert.deepEqual([...firsts].sort(), [1, 2]);
});
