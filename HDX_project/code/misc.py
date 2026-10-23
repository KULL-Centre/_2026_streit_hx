import pandas as pd

# function to read experimental data file with metadata records
# see examplefile_NMR.csv
# to load: df = read_data_NMR('examplefile_NMR.csv')
# see metadata: df.attrs

def read_data(filename):

    # check metadata entries
    with open(filename,'r') as f:
        lines=f.readlines()
        count=0
        for line in lines[1:]:
            if line[0]=='#':
                count+=1
            else:
                break

    # read data
    df = pd.read_csv(filename,skiprows=count+1)

    # add metadata
    attrs_dict = {}
    for i in range(1,count+1):
        key = lines[i][1:].strip().split(':')[0]
        entry = lines[i][1:].strip().split(':')[1]
        try:
            float(entry)
            #print("Metadata entry float")
            attrs_dict[key]=float(entry)
        except ValueError:
            #print("Metadata entry not a float")
            attrs_dict[key]=entry
    df.attrs = attrs_dict

    return df
