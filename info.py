import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import datetime

class Info():
    def __init__(self,args):
        self.title = 'Policy %s frame %d step %s max_latency %d beta %d lr %f tau %d' % (args.policy, args.num_frames, args.num_steps, args.max_latency, args.beta, args.lr, args.tau)
        if not os.path.exists('./result/'):
            os.mkdir('./result/')
        self.path = './result/'
        self.name = 'server-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.time_count = 0
        num_process = args.num_processes
        self.process_count = num_process

        self.time = [num_process*[]]
        self.num = [num_process * []]
        self.accu = [num_process*[]]
        self.len_q = [num_process*[]]
        self.batchsz = [num_process*[]]
        self.reward = [num_process*[]]
        self.overdue_ratio = [num_process*[]]
        self.overdue_rate = [num_process * []]
        self.arrive_rate = [num_process*[]]
        self.throughput = [num_process*[]]
        self.model_num = [num_process*[]]
        self.actions = [num_process*[]]
        self.wait = [num_process*[[],[],[],[],[],[]]]

    def insert(self, dict):
        for i,data in enumerate(dict):
            if data == None:
                break
            self.time[i].append(float(data['time']))
            self.num[i].append(int(data['num']))
            self.reward[i].append(float(data['reward']))
            self.accu[i].append(float(data['accu']))
            self.overdue_ratio[i].append(float(data['overdue'])/float(data['num']))
            self.overdue_rate[i].append(float(data['overdue_rate']))
            self.throughput[i].append(float(data['throughput']))
            self.arrive_rate[i].append(int(data['arrive_rate']))
            self.len_q[i].append(int(data['len_q']))
            self.batchsz[i].append(int(data['batchsz']))
            self.model_num[i].append(int(data['model_num']))
            self.actions[i].extend(data['actions'])
            for m in range(6):
                self.wait[i][m].extend(data['wait'][m])
            self.time_count += 1

    def show(self):
        for i in range(self.process_count):
            pic1_x = self.time[i]
            pic1_y1 = self.accu[i]
            pic1_y2 = self.arrive_rate[i]

            pic2_y1 = self.overdue_ratio[i]
            pic2_y2 = self.arrive_rate[i]

            pic3_y1 = self.reward[i]
            pic3_y2 = self.arrive_rate[i]

            pic4_y = self.wait[i]

            # 定义figure
            fig = plt.figure(figsize=(40, 12))
            plt.axis('off')
            plt.title(self.title, fontsize=20)

            ax1 = fig.add_subplot(321)
            p1, = ax1.plot(pic1_x, pic1_y1, label="ACCU", color="b", linestyle="-", marker="o", linewidth=1)

            ax2 = ax1.twinx()
            p2, = ax2.plot(pic1_x, pic1_y2, label="Requests", color="r", linestyle="-", marker="^", linewidth=1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('ACCU', color='b')
            ax2.set_ylabel('Requests', color='r')
            # 显示图示
            plt.legend()

            ax1 = fig.add_subplot(322)
            p1, = ax1.plot(pic1_x, pic2_y1, label="Overdue", color="b", linestyle="-", marker="o", linewidth=1)

            ax2 = ax1.twinx()
            p2, = ax2.plot(pic1_x, pic2_y2, label="Requests", color="r", linestyle="-", marker="^", linewidth=1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Overdue', color='b')
            ax2.set_ylabel('Requests', color='r')
            # 显示图示
            plt.legend()

            ax1 = fig.add_subplot(323)
            p1, = ax1.plot(pic1_x, pic3_y1, label="Reward", color="b", linestyle="-", marker="o", linewidth=1)

            ax2 = ax1.twinx()
            p2, = ax2.plot(pic1_x, pic3_y2, label="Requests", color="r", linestyle="-", marker="^", linewidth=1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Reward', color='b')
            ax2.set_ylabel('Requests', color='r')
            # 显示图示
            plt.legend()

            ax1 = fig.add_subplot(324)
            ax1.plot(pic1_x, pic4_y[0], label="model1", linestyle="-", marker="o", linewidth=1)
            ax1.plot(pic1_x, pic4_y[1], label="model2", linestyle="-", marker="o", linewidth=1)
            ax1.plot(pic1_x, pic4_y[2], label="model3", linestyle="-", marker="o", linewidth=1)

            ax1.set_xlabel('Time')
            ax1.set_ylabel('wait_time',)
            # 显示图示
            plt.legend()

            ax1 = fig.add_subplot(325)

            ax1.plot(pic1_x, pic4_y[3], label="max_queue_wait",linestyle="-", marker="^", linewidth=1)
            ax1.plot(pic1_x, pic4_y[4], label="max_process_wait",linestyle="-", marker="^", linewidth=1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('wait_time',)
            # 显示图示
            plt.legend()

            ax1 = fig.add_subplot(326)

            ax1.plot(pic1_x, pic4_y[5], label="latency",linestyle="-", marker="^", linewidth=1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('latency',)
            # 显示图示
            plt.legend()

            plt.show()

            name = self.name + "_%d.png" % i
            plt.savefig(self.path+name)
