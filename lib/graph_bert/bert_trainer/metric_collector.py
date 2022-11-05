from collections import defaultdict
from typing import Dict, List

TEST_METRICS = "test_metrics"
TRAIN_METRICS = "train_metrics"


class BatchMetric:
    def __init__(self, num: int = 0, value: float = 0):
        self.num = num
        self.value = value

    def update(self, num: int, value: float):
        weight = self.num * self.value + value

        self.num += num
        self.value = weight / self.num

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.__str__())


class MetricCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.batch_metrics = defaultdict(BatchMetric)

    def collect_batch(self, name: str, value: float, num: int):
        batch_metric = self.batch_metrics[name]
        batch_metric.update(num, value)
        self.batch_metrics[name] = batch_metric

    def update_metrics(self):
        for k, v in self.batch_metrics.items():
            self.metrics[k].append(v.value)

        self.batch_metrics = defaultdict(BatchMetric)

    def get_metrics(self) -> Dict[str, List]:
        return self.metrics


class MLMetricCollector:
    def __init__(self):
        self.train_metric_collector = MetricCollector()
        self.test_metric_collector = MetricCollector()

    def collect_train_batch(self, name: str, value: float, num: int):
        self.train_metric_collector.collect_batch(name=name, value=value, num=num)

    def collect_test_batch(self, name: str, value: float, num: int):
        self.test_metric_collector.collect_batch(name=name, value=value, num=num)

    def update_train_collector(self):
        self.train_metric_collector.update_metrics()

    def update_test_collector(self):
        self.test_metric_collector.update_metrics()

    def get_metrics(self) -> Dict[str, Dict[str, List]]:
        return {
            TEST_METRICS: self.test_metric_collector.get_metrics(),
            TRAIN_METRICS: self.train_metric_collector.get_metrics(),
        }
