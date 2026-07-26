"use strict";
// Minimal stub of the nats client surface argus-kengram-consumer imports.
// The truth-trio harness drives per-message handlers directly with fake msg
// objects — no live NATS server or subject is ever touched.
module.exports = {
  connect: async () => {
    throw new Error("nats stub: the test harness never opens a NATS connection");
  },
  AckPolicy: { Explicit: "explicit" },
  DeliverPolicy: { All: "all", StartTime: "by_start_time" },
  RetentionPolicy: { Limits: "limits" },
  StorageType: { File: "file" },
  DiscardPolicy: { Old: "old", New: "new" },
  nanos: (ms) => ms * 1e6,
  StringCodec: () => ({
    encode: (value) => Buffer.from(value, "utf8"),
    decode: (data) => Buffer.from(data).toString("utf8"),
  }),
};
