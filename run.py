import os

cmd_py = ''
root_file = ''
result = os.popen("uname -a").read()
print(result)

if result[:5] == 'Linux':
    # linux
    cmd_py = "python "
    root_file = "/hdd2/qingpeng/a2c-master/"
elif result[:6] == 'Darwin':
    # mac
    cmd_py = "/usr/local/bin/python3.6 "
    root_file = os.path.dirname(__file__) + '/'
else:
    raise NotImplementedError
print(root_file)


# for i in [1,2,3,5,8]:
#     for j in [3,5,8,10,15]:
#         print(cmd_py + root_file + "main.py --cuda --num-steps %d --max_latency %d --num-frames 80000" % (i, j))
#         os.system(cmd_py + root_file + "main.py --cuda --num-steps %d --max_latency %d --num-frames 80000" % (i, j))

# for i in [15,27,48,64]:
#     for j in [8,10,15,20]:
#         print(cmd_py + root_file + "main.py --cuda --num-steps %d --max_latency %d --num-frames 80000" % (i, j))
#         os.system(cmd_py + root_file + "main.py --cuda --num-steps %d --max_latency %d --num-frames 80000" % (i, j))


for i in [5,16,27,64]:
    for j in [2,3,4,5,6]:
        print(cmd_py + root_file + "main.py --cuda --num-steps %d --beta 0 --tau %d --num-frames 80000" % (i, j))
        os.system(cmd_py + root_file + "main.py --cuda --num-steps %d --beta 0 --tau %d --num-frames 80000" % (i, j))