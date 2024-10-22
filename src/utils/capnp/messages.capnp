@0xd8bfb47e358210a0;

struct DriverPacket {
  channelSamples @0 :List(SamplesBuffer);
  sqnNum @1 :UInt32;
  rxRate @2 :Float64;
  txRate @3 :Float64;
  rxCtrFreq @4 :Float64;
  txCtrFreq @5 :Float64;
  numRxSamples @6 :UInt32;
  sqnTime @7 :Float64;
  timeToSendSamples @8 :Float64;
  startBurst @9 :Bool;  # start of burst flag
  endBurst @10 :Bool; # end of burst flag
  alignSqns @11 :Bool;

  struct SamplesBuffer {
    real @0 :List(Float32);
    imag @1 :List(Float32);
  }
}

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
