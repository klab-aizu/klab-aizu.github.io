`timescale 1ns/1ps

module tb_top;

reg a;
reg b;
reg c;

wire y;

top dut (
    .a(a),
    .b(b),
    .c(c),
    .y(y)
);

initial begin

    a=0; b=0; c=0;
    #10;

    a=1; b=0; c=0;
    #10;

    a=1; b=1; c=0;
    #10;

    a=0; b=0; c=1;
    #10;

    a=1; b=1; c=1;
    #10;

    $finish;

end

endmodule