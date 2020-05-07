from linguistic_and_stylistic_complexity.bin import lascomplexity as ssc
mypath = 'ssc_data'
cnt = 1
for filename in os.listdir(mypath):
    print(f'Now processing text: {filename}, text {cnt} of {len(os.listdir(mypath))}')
    cnt += 1

    ssc.arguments()
    ssc.main()
