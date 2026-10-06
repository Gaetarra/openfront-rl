import assert from "node:assert/strict";
import { test } from "node:test";
import { UnitType } from "../OpenFront_Source_Code/openfrontio/src/core/game/Game";
import { Act, attackMask, decodeAction, typeMask } from "./actions";
import { EnvGame } from "./game";
import { ONION, script, spawnCell } from "./testkit";

const ticks = (env: EnvGame, n: number) => {
  for (let i = 0; i < n; i++) env.tickWith([]);
};

test("spawn is open early, forced late and closed at the very end of the spawn phase", async () => {
  const env = await EnvGame.create(ONION, "masks");
  const seat = env.seats[0];
  const types = () => [...typeMask(env.game, seat, attackMask(env.game, seat, env.slots))];
  ticks(env, 100);
  assert.deepEqual(types(), [1, 1, 0, 0, 0]);
  ticks(env, 85); // tick 185
  assert.deepEqual(types(), [0, 1, 0, 0, 0]);
  ticks(env, 12); // tick 197
  assert.deepEqual(types(), [1, 0, 0, 0, 0]);
});

test("spawn decodes to unowned land inside the chosen cell", async () => {
  const env = await EnvGame.create(ONION, "spawn");
  const seat = env.seats[0];
  const cell = spawnCell(env);
  const types = typeMask(env.game, seat, attackMask(env.game, seat, env.slots));
  const { intent, valid } = decodeAction(env.game, seat, env.frame, types, new Uint8Array(env.slots), [Act.Spawn, cell, 0, 0]);
  assert.ok(valid && intent !== null && intent.type === "spawn");
  if (intent?.type !== "spawn") return;
  assert.equal(env.frame.cellOf(intent.tile), cell);
  assert.ok(env.game.isLand(intent.tile) && !env.game.hasOwner(intent.tile));
});

test("after spawning, unowned land is attackable and the troop share sets the troops", async () => {
  const env = await EnvGame.create(ONION, "attack");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  assert.equal(attack[0], 1);
  const types = typeMask(env.game, seat, attack);
  assert.equal(types[Act.Attack], 1);
  const { intent, valid } = decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, 0, 37]);
  assert.ok(valid);
  assert.deepEqual(intent, { type: "attack", targetID: null, troops: 0.37 * seat.troops() });
  const far = [...attack.keys()].find((s) => s > 1 && attack[s] === 0);
  if (far !== undefined) {
    assert.equal(decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, far, 10]).valid, false);
  }
});

test("the five sizes of earlier networks send what they sent before", async () => {
  const env = await EnvGame.create(ONION, "sizes");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  const types = typeMask(env.game, seat, attack);
  for (const [share, fraction] of [[10, 0.1], [25, 0.25], [50, 0.5], [75, 0.75], [100, 1.0]] as const) {
    const { intent } = decodeAction(env.game, seat, env.frame, types, attack, [Act.Attack, 0, 0, share]);
    assert.deepEqual(intent, { type: "attack", targetID: null, troops: fraction * seat.troops() }, `${share}%`);
  }
});

test("build city: valid on own land with gold, dropped elsewhere", async () => {
  const env = await EnvGame.create(ONION, "city");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(10_000_000n);
  const attack = attackMask(env.game, seat, env.slots);
  const types = typeMask(env.game, seat, attack);
  assert.equal(types[Act.BuildCity], 1);
  const own = env.frame.cellOf([...seat.tiles()][0]);
  const ok = decodeAction(env.game, seat, env.frame, types, attack, [Act.BuildCity, own, 0, 0]);
  assert.equal(ok.valid, true);
  assert.equal(ok.intent?.type === "build_unit" && ok.intent.unit, UnitType.City);
  assert.equal(decodeAction(env.game, seat, env.frame, types, attack, [Act.BuildCity, 0, 0, 0]).valid, false);
});

test("malformed or unusable actions are dropped, never thrown", async () => {
  const env = await EnvGame.create(ONION, "robust");
  script(env, 260);
  const seat = env.seats[0];
  const attack = attackMask(env.game, seat, env.slots);
  const types = typeMask(env.game, seat, attack);
  const bad = [[9, 0, 0, 0], [Act.Attack, 0, 99, 0], [Act.Attack, 0, 0, 0], [Act.Attack, 0, 0, 101], [Act.Attack, 0, 0, 2.5], [Act.Boat, 0, 0, 0], [Act.Spawn, 0, 0, 0], [Act.BuildCity, -1, 0, 0], [Act.BuildCity, 1e9, 0, 0], []];
  for (const a of bad) assert.equal(decodeAction(env.game, seat, env.frame, types, attack, a).valid, false, `${a}`);
  for (let c = 0; c < env.frame.n; c += 7) decodeAction(env.game, seat, env.frame, types, attack, [Act.Boat, c, 0, 25]);
});
