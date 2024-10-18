@0x8156beb63a2eaed2;

struct RxSamplesMetadata {
  sqnNum @0 :UInt32;
  numRxSamples @1 :UInt32;
  rxRate @2 :Float64;
  sqnTime @3 :Float64;
  initTime @4 :Float64;
  sqnStartTime @5 :Float64;
  ringbufferSize @6 :UInt64;
  agcStatusH @7 :UInt32;
  lpStatusH @8 :UInt32;
  agcStatusL @9 :UInt32;
  lpStatusL @10 :UInt32;
  gpsLocked @11 :Bool;
  gpsToSystemTimeDiff @12 :Float64;
}
