@0xd8bfb47e358210a0;

struct DriverPacketPnp {
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
