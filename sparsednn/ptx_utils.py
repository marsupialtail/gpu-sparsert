from utils import half_to_hex

def parse_ptx(filename,A_blocks):
    ptx = open(filename,"r").readlines()
    reg_names = []
    global_address_names = []
    #saved_global_address_name = None
    for block in range(A_blocks):
        block_reg_names = []
        look_for = "B" + str(block) + "G0"
        for line_num in range(len(ptx)):
            if look_for in ptx[line_num]:
                for j in range(3,10000):
                    if "fma.rn.f32" in ptx[line_num+j]:
                        register_name = ptx[line_num+j].split()[1].replace(",","")
                        block_reg_names.append(int(register_name.replace("%f","")))
                    if "END" in ptx[line_num+j]:
                        break
                #print(line_num)
                global_address_name = None
                for j in range(2,10000):
                    if "ld.global" in ptx[line_num + j]:
                        if "+" in ptx[line_num + j]:
                            global_address_name = ptx[line_num+j].split("[")[1].split("+")[0]
                        else:
                            global_address_name = ptx[line_num+j].split("[")[1].split("]")[0]
                        break
                if global_address_name is None:
                    print("Can't detect global address register name.")
                    raise Exception

                global_address_names.append(global_address_name)
                reg_names.append(block_reg_names)

    #print(reg_names)
    return reg_names, global_address_names

import random

HALF_BIAS_RELU_BB="""
    mov.b32 bias_reg, BIAS;
    add.f16x2 temp_reg, SOURCE, bias_reg;
    set.gt.ftz.f16x2.f16x2 pred_reg, temp_reg, zero_reg;
    mul.rn.f16x2 DEST, pred_reg, temp_reg;
"""

HALF_BIAS_BB="""
    mov.b32 bias_reg, BIAS;
    add.f16x2 DEST, SOURCE, bias_reg;
"""

import struct
def hex_to_float(hex):
    return struct.unpack('!f', bytes.fromhex(hex))[0]

def insert_ptx(in_ptx_file,out_ptx_file, block_ptxs,relu=True,blurb=None,id = None):
    ptx_code = open(in_ptx_file,"r").readlines()
    new_file = open(out_ptx_file,"w")
    i = 0
    mads = 0
    while i < len(ptx_code):
        line = ptx_code[i]
        #print("change ptx utils line 64 for your problems")
        if blurb and "mad.lo.s32" in line and " " + str(id ) + "," in line:
            if mads == 0:
                stuff = line.replace("\n","").split(",")
                x_reg = stuff[1]
                y_reg = stuff[3].replace(";","")
                new_file.write(line)
                new_file.write(blurb.replace("X_REG",x_reg).replace("Y_REG",y_reg))
            else:
                new_file.write(line)
            mads += 1
        elif "START" in line:
            my_block = int(line.split("B")[1].split("G")[0])
            my_group = int(line.split("G")[1].split(";")[0])
            my_ptx = block_ptxs[my_block][my_group]

            # we have to deal with cases where the load instruction doesn't
            # immediately follow the inline assembly marker
            # we can't miss those instructions in between!
            new_file.write(line)

            for j in range(i+2,len(ptx_code)):
                if "ld.global" in ptx_code[j]:
                    break
                else:
                    new_file.write(ptx_code[j])

            for ptx_line in my_ptx:
                new_file.write(ptx_line)

            new_file.write("\n")
            j = i
            next_i = None
            while True:
                if "END" in ptx_code[j]:
                    next_i = j + 1
                    break
                j += 1

            i = next_i

        # TODO: this is really hacky, you should not be using this.
        # we are parsing the ptx, finding the part where we add bias and perform relu
        # and manually swapping out that block for a handcoded ptx block that does the equivalent.
        # we literally parse the hex literal bias value, convert it to float, then to float16, then to hex again
        # we should change this approach.
        elif "add.f32" in line and "mov.f32" in ptx_code[i+1] and "max.f32" in ptx_code[i+2]:
            SOURCE = line.split(",")[-2]
            DEST = ptx_code[i+2].split(",")[-3].split()[-1]
            BIAS = half_to_hex(hex_to_float(line.split(",")[-1].split(";")[0].replace("0f","").replace(" ","")))
            if relu:
                new_file.write(HALF_BIAS_RELU_BB.replace("SOURCE",SOURCE).replace("DEST",DEST).replace("BIAS",BIAS))
            else:
                new_file.write(HALF_BIAS_BB.replace("SOURCE",SOURCE).replace("DEST",DEST).replace("BIAS",BIAS))
            i += 2
        elif "add.f32" in line and "max.f32" in ptx_code[i+1]:
            SOURCE = line.split(",")[-2]
            DEST = ptx_code[i+1].split(",")[-3].split()[-1]
            BIAS = half_to_hex(hex_to_float(line.split(",")[-1].split(";")[0].replace("0f","").replace(" ","")))
            if relu:
                new_file.write(HALF_BIAS_RELU_BB.replace("SOURCE",SOURCE).replace("DEST",DEST).replace("BIAS",BIAS))
            else:
                new_file.write(HALF_BIAS_BB.replace("SOURCE",SOURCE).replace("DEST",DEST).replace("BIAS",BIAS))
            i += 1
        else:
            new_file.write(line)

        i += 1
