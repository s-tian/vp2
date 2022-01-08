import socket
import struct
import numpy as np

from vp2.mpc.agent import Agent
from vp2.mpc.utils import ObservationList


class RemoteAgentWrapper(Agent):
    """
    Wrap an agent class so that it communicates with the environment through
    sockets. Lets us communicate between ROS python2 environment and python3
    control code, and also offload computation to the cloud.
    """

    def __init__(self, agent, host, port):
        self._agent = agent
        self.host = host
        self.port = port
        self.socket, self.conn = self.setup_socket()
        self.num_traj = 0
        self.obs_history = None

    def setup_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(
            "Setting server to accept client connections at {}:{}".format(
                self.host, self.port
            )
        )
        sock.bind((self.host, self.port))
        sock.listen()
        conn, addr = sock.accept()
        print(f"Connected by {addr}")
        return sock, conn

    def send_bytes(self, s):
        self.conn.sendall(struct.pack("I", len(s)))
        self.conn.sendall(s)

    def send_np_array(self, a):
        self.conn.sendall(struct.pack("I", len(a.shape)))
        [self.conn.sendall(struct.pack("I", d)) for d in a.shape]
        self.conn.sendall(a.astype(np.float32).tobytes())

    def _receive_data_size(self):
        size_bytes = self._receive_bytes(4)
        size = struct.unpack("I", size_bytes)[0]
        return size

    def _receive_message_string(self):
        size = self._receive_data_size()
        msg = self._receive_bytes(size)
        return msg

    def _receive_message_np(self):
        shape_len = self._receive_data_size()
        shape = [self._receive_data_size() for _ in range(shape_len)]
        msg = self._receive_bytes(np.prod(shape) * 4)
        array = np.frombuffer(msg, dtype=np.float32).reshape(shape)
        return array

    def _receive_bytes(self, msg_size):
        """
        :param msg_size: Exact number of bytes to read from connection
        :return: message in bytes
        """
        msg = bytes()
        while msg_size > 0:
            new_part = self.conn.recv(msg_size)
            msg_size -= len(new_part)
            msg += new_part
        return msg

    def get_message_dict(self):
        d = None
        message_complete = False
        while not message_complete:
            key = self._receive_message_string().decode()
            # print('Received key ', key)
            if key == "SOM":
                d = dict()
            elif key == "EOM":
                message_complete = True
            else:
                data = self._receive_message_np()
                d[key] = data
        # print(d)
        return d

    def listen(self):
        while True:
            d = self.get_message_dict()
            # for k, v in d.items():
            #     print(k)
            #     print(v.shape)
            d["t"] = int(d["t"].item())  # Cast the timestep t to an integer
            if d["t"] == 0:
                if self.obs_history is not None:
                    self.obs_history.append(
                        self._agent.goal.repeat(len(self.obs_history))
                    )
                    self.obs_history.log_gif(f"{self._agent.log_dir}/traj_vis")
                    self.obs_history = None
                self._agent.reset()
                self._agent.set_log_dir(f"./traj_{self.num_traj}/")
                self.num_traj += 1
            image_size = d["images"].shape[-3:-1]
            goal_image = ObservationList(
                dict(rgb=d["goal_image"]),
                add_time_dimension=True,
                image_shape=image_size,
            )
            obs_history = ObservationList(
                dict(rgb=d["images"] / 255.0),
                add_time_dimension=False,
                image_shape=image_size,
            )
            obs_history.log_gif(f"{self._agent.log_dir}/step_{d['t']}_partial")
            self.obs_history = obs_history
            self._agent.set_goal(goal_image)
            goal_image.save_image(f"{self._agent.log_dir}/goal_image")

            action = self._agent.act(
                t=d["t"], obs_history=obs_history, state_obs_history=d["state"]
            )
            print(f"Sending action {action}")
            self.send_np_array(action.astype(np.float32))

    def act(self, t, obs_history, act_history):
        return self._agent.act(t, obs_history, act_history)


if __name__ == "__main__":
    agent = RemoteAgentWrapper(Agent(), "", 65433)
    agent.listen()
