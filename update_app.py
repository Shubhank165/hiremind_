import sys

def modify_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # We want to find the line: if mode in ["conversation", "planning", "init", "profiling"]:
    # and replace it with:
    # if mode in ["conversation", "planning", "init", "profiling", "workspace"]:
    #     left_col, right_col = st.columns([1, 1])
    #     with left_col:
    # Then indent everything until elif mode == "workspace": 
    # Change elif mode == "workspace": to with right_col:
    # Then indent the rest until elif mode == "complete":
    
    out = []
    in_left = False
    in_right = False
    
    for line in lines:
        if 'if mode in ["conversation", "planning", "init", "profiling"]:' in line:
            out.append('    if mode in ["conversation", "planning", "init", "profiling", "workspace"]:\n')
            out.append('        left_col, right_col = st.columns([1, 1.2])\n')
            out.append('        with left_col:\n')
            in_left = True
            continue
            
        if 'elif mode == "workspace":' in line:
            in_left = False
            out.append('        with right_col:\n')
            in_right = True
            continue
            
        if 'elif mode == "complete":' in line:
            in_right = False
            out.append(line)
            continue
            
        if in_left or in_right:
            if line.strip() == "":
                out.append("\n")
            else:
                out.append("    " + line)
        else:
            out.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(out)

modify_file("/home/shubhank165/VSCode/eightfold/frontend/app.py")
