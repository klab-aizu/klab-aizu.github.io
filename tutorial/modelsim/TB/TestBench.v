module TestBench;

  reg CLK;
  reg RESET;
  reg D;

  wire D;

  Flip_Flop uut (
    .CLK    (CLK    ),
    .RESET  (RESET  ),
    .D      (D      ),
    .Q      (Q      )
  );

  always
    #5 CLK =~CLK;

  initial begin
    CLK = 0;
    RESET = 0;
    D = 0;

    #50

    #3 D = 1'b1;
    #5 D = 1'b0;
    #10 D = 1'b1;
    #12 RESET = 1'b1;
    #10 RESET = 1'b0;
    #10 $stop;
  end

endmodule

