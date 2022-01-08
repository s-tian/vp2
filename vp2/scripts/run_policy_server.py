import socket
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate

from vp2.util.remote_agent_wrapper import RemoteAgentWrapper


def get_server_hostname():
    return socket.gethostbyname(socket.gethostname())


@hydra.main(config_path="configs", config_name="policy_server_config")
def start_server(cfg):
    with open("config.yaml", "w") as f:
        OmegaConf.save(config=cfg, f=f.name)
    # print(cfg.pretty())
    model = instantiate(cfg.model)
    server_hostname = get_server_hostname()
    agent = RemoteAgentWrapper(
        instantiate(cfg.agent, optimizer={"model": model}),
        host=server_hostname,
        port=cfg.port,
    )
    agent.listen()


if __name__ == "__main__":
    start_server()
