import sys

infile = sys.argv[1]
outfile = open(sys.argv[2],"w")

lines = open(infile).readlines()
flag = 0
for line in lines:
    if "fma.rn.f32" in line and "0f" in line.split(",")[2]:
        constant = line.split(",")[2]
        if flag == 0:
            outfile.write(".reg .f32 temp_reg;\n")
            outfile.write(".reg .f32 virg_reg;\n")
            flag = 1
        else:
            pass

        if "0f00000000" in line.split(",")[3]:
            outfile.write("mov.b32 virg_reg," + "0x00000000" + ";\n")
            outfile.write("mov.b32 temp_reg," + constant + ";\n")
            outfile.write(line.replace(constant,"temp_reg").replace("0f00000000","virg_reg"))
        else:
            outfile.write("mov.b32 temp_reg," + constant + ";\n")
            outfile.write(line.replace(constant,"temp_reg"))
    else:
        outfile.write(line)
