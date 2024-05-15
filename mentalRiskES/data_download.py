import requests, zipfile, io
from requests.adapters import HTTPAdapter, Retry
from typing import List, Dict
import random
import json
import os
import pandas as pd
from codecarbon import EmissionsTracker
import numpy as np
from transformers import BertModel, BertTokenizer

URL = "http://s3-ceatic.ujaen.es:8036"
TOKEN = "329a4038d2e5bc75b9dc592ddcea08da9298c7ae"

# Download endpoints
ENDPOINT_DOWNLOAD_TRIAL = URL + "/{TASK}/download_trial/{TOKEN}"
ENDPOINT_DOWNLOAD_TRAIN = URL + "/{TASK}/download_train/{TOKEN}"

# Trial endpoints
ENDPOINT_GET_MESSAGES_TRIAL = URL + "/{TASK}/getmessages_trial/{TOKEN}"
ENDPOINT_SUBMIT_DECISIONS_TRIAL = URL + "/{TASK}/submit_trial/{TOKEN}/{RUN}"

# Test endpoints
ENDPOINT_GET_MESSAGES = URL + "/{TASK}/getmessages/{TOKEN}"
ENDPOINT_SUBMIT_DECISIONS = URL + "/{TASK}/submit/{TOKEN}/{RUN}"


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def download_messages_trial(task: str, token: str):
    """ Allows you to download the trial data of the task.
        Args:
          task (str): task from which the data is to be retrieved
          token (str): authentication token
    """

    response = requests.get(ENDPOINT_DOWNLOAD_TRAIN.format(TASK=task, TOKEN=token))

    if response.status_code != 200:
        print("Trial - Status Code " + task + ": " + str(response.status_code) + " - Error: " + str(response.text))
    else:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        os.makedirs("./data/{task}/trial/".format(task=task))
        z.extractall("./data/{task}/trial/".format(task=task))


class Client_taskX:
    """ Client communicating with the official server.
        Attributes:
            task (str): task in which you wish to participate
            token (str): authentication token
            number_of_runs (int): number of systems. Must be 3 in order to advance to the next round.
            tracker (EmissionsTracker): object to calculate the carbon footprint in prediction

    """

    def __init__(self, task: str, token: str, number_of_runs: int, tracker: EmissionsTracker):
        self.task = task
        self.token = token
        self.number_of_runs = number_of_runs
        self.tracker = tracker
        # Required parameters
        self.relevant_cols = [
            "duration", "emissions", "cpu_energy", "gpu_energy", "ram_energy",
            "energy_consumed", "cpu_count", "gpu_count", "cpu_model", "gpu_model",
            "ram_total_size", "country_iso_code"
        ]

    def get_messages(self, retries: int, backoff: float) -> Dict:
        """ Allows you to download the test data of the task by rounds.
            Here a GET request is sent to the server to extract the data.
            Args:
              retries (int): number of calls on the server connection
              backoff (float): time between retries
        """
        session = requests.Session()
        retries = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))

        response = session.get(
            ENDPOINT_GET_MESSAGES_TRIAL.format(TASK=self.task, TOKEN=self.token))  # ENDPOINT FOR TRIAL

        if response.status_code != 200:
            print("GET - Task {} - Status Code {} - Error: {}".format(self.task, str(response.status_code),
                                                                      str(response.text)))
            return []
        else:
            return json.loads(response.content)

    def submit_decission(self, messages: List[Dict], emissions: Dict, retries: int, backoff: float):
        """ Allows you to submit the decisions of the task by rounds.
            The POST requests are sent to the server to send predictions and carbon emission data
            Args:
              messages (List[Dict]): Message set of the current round
              emissions (Dict): carbon footprint generated in the prediction
              retries (int): number of calls on the server connection
              backoff (float): time between retries
        """
        decisions = {}
        contexts = {}
        labels_task1_list = ['none', 'depression', 'anxiety']  # Example for task1 and task2
        labels_task2_list = ["addiction", "emergency", "family", "work", "social", "other", "none"]  # Example for task2

        # You must create the appropriate structure to send the predictions according to each task
        for message in messages:
            decisions[message["nick"]] = random.choice(labels_task1_list)
            contexts[message["nick"]] = random.choice(labels_task2_list) + "#" + random.choice(labels_task2_list)

        data_task1 = {
            "predictions": decisions,
            "emissions": emissions
        }
        data_task2 = {
            "predictions": decisions,
            "contexts": contexts,
            "emissions": emissions
        }
        data_task1 = json.dumps(data_task1, ensure_ascii=False, default=default_dump)
        data_task2 = json.dumps(data_task2, ensure_ascii=False, default=default_dump)

        # Session to POST request
        session = requests.Session()
        retries = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        data = {'task1': data_task1, 'task2': data_task2}
        for item in ['task1', 'task2']:
            for run in range(0, self.number_of_runs):
                # For each run, new decisions
                response = session.post(
                    ENDPOINT_SUBMIT_DECISIONS_TRIAL.format(TASK=item, TOKEN=self.token, RUN=run),  # 改(TASK=self.task,
                    json=[data[item]])  # ENDPOINT FOR TRIAL

                if response.status_code != 200:
                    print("POST - Task {} - Status Code {} - Error: {}".format(item, str(response.status_code),
                                                                               str(response.text)))
                    return
                else:
                    print("POST - Task {} - run {} - Message: {}".format(item, run, str(response.text)))

    def run_taskX(self, retries: int, backoff: float):
        """ Main thread
            Args:
              retries (int): number of calls on the server connection
              backoff (float): time between retries
        """
        # Get messages for taskX
        messages = self.get_messages(retries, backoff)

        # If there are no messages
        if len(messages) == 0:
            print("All rounds processed")
            return

        while len(messages) > 0:  # 无限循环直至结束
            print("------------------- Processing round {}".format(messages[0]["round"]))
            print(messages)
            # Save subjects
            with open('./data/rounds_trial/round{}.json'.format(messages[0]["round"]), 'w+',
                      encoding='utf8') as json_file:
                json.dump(messages, json_file, ensure_ascii=False)  # 将message的信息存入json_file

            # Calculate emissions for each prediction
            self.tracker.start()

            # Your code
            tokenizer = BertTokenizer.from_pretrained(r'D:\nlp\bert\bert-base-spanish-wwm-uncased')
            bert_model = BertModel.from_pretrained(r'D:\nlp\bert\bert-base-spanish-wwm-uncased')

            emissions = self.tracker.stop()
            df = pd.read_csv("emissions.csv")
            measurements = df.iloc[-1][self.relevant_cols].to_dict()
            # temp_measurements = measurements
            # for item in self.relevant_cols:
            #     if type(temp_measurements[item]) == np.int64:
            #         measurements[item] = int(temp_measurements[item])
            self.submit_decission(messages, measurements, retries, backoff)

            # One GET request for each round
            messages = self.get_messages(retries, backoff)

        print("All rounds processed")


def download_data():
    download_messages_trial("task1", TOKEN)


def get_post_data():
    # Emissions Tracker Config
    config = {
        "save_to_file": True,
        "log_level": "DEBUG",
        "tracking_mode": "process",
        "output_dir": ".",
    }
    tracker = EmissionsTracker(**config)

    number_runs = 3  # Max: 3

    # Prediction period
    client_taskX = Client_taskX("task1", TOKEN, number_runs, tracker)
    client_taskX.run_taskX(5, 0.1)


if __name__ == '__main__':
    download_data()

    # get_post_data()
    # exit(0)
    # config = {
    #     "save_to_file": True,
    #     "log_level": "DEBUG",
    #     "tracking_mode": "process",
    #     "output_dir": ".",
    # }
    #
    # tracker = EmissionsTracker(**config)
    # client_task1 = Client_taskX("task1", TOKEN, 5, tracker)
    # res = client_task1.get_messages(5, 0.1)
    # print(res)
