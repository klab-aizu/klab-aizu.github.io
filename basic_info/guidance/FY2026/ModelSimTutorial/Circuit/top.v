module top(
    input  a,
    input  b,
    input  c,
    output y
);

wire and_out;

and_gate u_and (
    .a(a),
    .b(b),
    .y(and_out)
);

or_gate u_or (
    .a(and_out),
    .b(c),
    .y(y)
);

endmodule