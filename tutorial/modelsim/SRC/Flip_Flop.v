module Flip_Flop(
  input CLK,
  input RESET,
  input D,
  output Q
);

  reg ff;
  assign Q = ff;

  always @(posedge CLK)
  begin
    if (RESET)
      ff <= 0;
    else
      ff <= D;
  end
endmodule

