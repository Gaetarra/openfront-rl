import assert from "node:assert/strict";
import { test } from "node:test";
import { attackMask } from "./actions";
import { EnvGame } from "./game";
import { encodeSeat, PLAYER_FEATURES, Rel, SCALARS } from "./obs";
import { ONION, script } from "./testkit";

test("seat view: self, enemies, unowned land row and bounded scalars", async () => {
  const env = await EnvGame.create(ONION, "obs");
  script(env, 260);
  const seat = env.seats[0];
  const P = env.slots;
  const players = new Float32Array(P * PLAYER_FEATURES);
  const rel = new Uint8Array(P);
  const scalars = new Float32Array(SCALARS);
  const attack = attackMask(env.game, seat, P);
  encodeSeat(env.game, seat, env.timerMinutes, env.frame.cs, attack, players, rel, scalars);
  const me = seat.smallID();
  assert.equal(rel[me], Rel.Self);
  assert.deepEqual([...rel].filter((r) => r === Rel.Enemy).length, 3);
  assert.equal(players[me * PLAYER_FEATURES + 2], 1);
  assert.ok(players[7] > 0 && players[7] <= 1, "unowned land share");
  assert.equal(players[12], attack[0]);
  assert.equal(scalars[3], 1, "spawned");
  for (const v of [...players, ...scalars]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 2, `${v}`);
});
