from libcity.data.dataset.downstream_task.downstream_dataset import DownstreamDataset


class LinearETADataset(DownstreamDataset):
    def process_traj(self, traj, tlist, max_len):
        if len(traj) <= max_len:
            last_traj = traj[-1]
            last_time = tlist[-1]
            traj = traj[:-1]
            tlist = tlist[:-1]
        else:
            last_traj = traj[max_len - 1]
            last_time = tlist[max_len - 1]
            traj = traj[:max_len - 1]
            tlist = tlist[:max_len - 1]
        return last_traj, last_time, traj, tlist

