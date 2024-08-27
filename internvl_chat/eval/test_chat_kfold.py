from collections import defaultdict

from internvl_chat.eval.test_chat_generic import test_model


VAL_META_PATH = 'shell/data/behavioral_val_kfold.json'


def test_chat_kfold():
    path_extensions = ['split1', 'split2', 'split3', 'split4', 'split5']
    # path_extensions = ['split1']
    # path_base = '/home/dvd/data/outputs/phq9_binary_on_vanilla_26B_lora'
    path_base = '/home/dvd/data/outputs/phq9_26B_lora'
    paths = [f'{path_base}_{ext}' for ext in path_extensions]
    # paths = ['/home/dvd/data/outputs/all_public_backbones_26B/checkpoint-5000'] * 5
    print(f'Paths: {paths}')
    dataset_base = 'behavioral_phq'
    datasets = [f'{dataset_base}_{ext}' for ext in path_extensions]
    metrics_list = []
    counts = []
    for path, dataset in zip(paths, datasets):
        metrics, count = test_model(meta_path=VAL_META_PATH,
                                    dataset_name=dataset,
                                    path=path)
        metrics_list.append(metrics)
        counts.append(count)

    metrics_aggregated = {}
    for metric_name in metrics_list[0].keys():
        metric_values = [metric[metric_name] for metric in metrics_list]
        metric_mean = sum(metric_values) / len(metric_values)
        metrics_aggregated[metric_name] = metric_mean
    total_counts = defaultdict(int)
    for count in counts:
        for key, value in count.items():
            total_counts[key] += value

    print('Individual metrics:', metrics_list)
    print('Individual counts:', counts)

    print('Aggregated metrics:', metrics_aggregated)
    print('Aggregated counts:', total_counts)
    return metrics_aggregated


if __name__ == '__main__':
    test_chat_kfold()
