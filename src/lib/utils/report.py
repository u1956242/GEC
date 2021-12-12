def report(dtgen, predicts, metrics, total_time, plus=""):
    """Calculate and organize metrics and predicts informations"""

    e_corpus = "\n".join([
        f"Total test sentences: {dtgen.size['test']}",
        f"{plus}",
        f"Total time:           {total_time}",
        f"Time per item:        {total_time / dtgen.size['test']}\n",
        f"Metrics (before):",
        f"Character Error Rate: {metrics[0][0]:.8f}",
        f"Word Error Rate:      {metrics[0][1]:.8f}",
        f"Sequence Error Rate:  {metrics[0][2]:.8f}\n",
        f"Metrics (after):",
        f"Character Error Rate: {metrics[1][0]:.8f}",
        f"Word Error Rate:      {metrics[1][1]:.8f}",
        f"Sequence Error Rate:  {metrics[1][2]:.8f}"
    ])

    p_corpus = []
    for i in range(dtgen.size['test']):
        p_corpus.append(f"GT {dtgen.dataset['test']['gt'][i]}")
        p_corpus.append(f"DT {dtgen.dataset['test']['dt'][i]}")
        p_corpus.append(f"PD {predicts[i]}\n")

    return (p_corpus, e_corpus)