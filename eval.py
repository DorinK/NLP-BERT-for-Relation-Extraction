import sys
from collections import defaultdict

GOLD_FILE = sys.argv[1]
PREDS_FILE = sys.argv[2]

WORK_FOR = "Work_For"


def extract_relations(path):

    id2relations = defaultdict(list)

    # Read the file.
    with open(path) as file:
        lines = file.readlines()

    for line in lines:  # For each annotation in the file
        sent_id, subject, relation, object, sentence = line.split("\t")

        if relation == WORK_FOR:  # Keep only the annotations that have a Work_For relation.
            id2relations[sent_id].append((subject, object, sentence))

    return id2relations


def regular_score(gold, preds):

    # Tracking variables
    hits = 0
    misses = []

    for sent_id, predicted_relations in preds.items():
        for (per_pred, org_pred, _) in predicted_relations:

            found = False  # Indicator flag

            if sent_id in gold:
                sent_gold_relations = gold[sent_id]

                # For each annotation under the same sentence id as the prediction
                for (per_gold, org_gold, sent) in sent_gold_relations:

                    # Search for a match while ignoring missing or extra determiners.
                    if per_gold == per_pred and (org_gold == org_pred or org_gold == 'the ' + org_pred or 'the ' +
                                                 org_gold == org_pred or org_gold == 'The ' + org_pred or 'The ' +
                                                 org_gold == org_pred):
                        found = True
                        hits += 1
                        # Remove this annotation to avoid from counting duplicates as hits.
                        gold[sent_id].remove((per_gold, org_gold, sent))
                        break

            if not found:  # If the prediction didn't appear in the annotations.
                misses.append((sent_id, per_pred, org_pred))

    return hits, misses


def generalized_score(gold, hits, misses):

    miss = []

    for (sent_id, per_pred, org_pred) in misses:

        found = False  # Indicator flag

        if sent_id in gold:
            sent_gold_relations = gold[sent_id]

            # For each annotation under the same sentence id as the prediction
            for (per_gold, org_gold, sent) in sent_gold_relations:

                # Search for a match
                if (per_gold in per_pred or per_pred in per_gold) and (org_gold in org_pred or org_pred in org_gold):
                    found = True
                    hits += 1
                    # Remove this annotation to avoid from counting duplicates as hits.
                    gold[sent_id].remove((per_gold, org_gold, sent))
                    break

        if not found:  # If the prediction didn't appear in the annotations.
            miss.append((per_pred, org_pred))

    return hits, miss


def calculate_score(gold, preds, hits, misses):

    # Calculate scores
    precision = round((hits / preds) * 100, 2) if preds != 0 else 0
    recall = round((hits / gold) * 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) != 0 else 0

    # Report scores
    print(f'Hit: {hits}')
    print(f'Miss: {len(misses)}')
    print(f"Precision: {precision}%")
    print(f"Recall: {recall}%")
    print(f"F1: {f1}%")


def eval_score(gold_file, preds_file, FULL_STATS=False):

    # Get the relevant Work_For relations from both the annotations and predictions file.
    annotations = extract_relations(gold_file)
    predictions = extract_relations(preds_file)

    total_gold = sum([len(sent_annotations) for sent_annotations in annotations.values()])
    total_preds = sum([len(sent_annotations) for sent_annotations in predictions.values()])
    print(f'Total Gold: {total_gold}')
    print(f'Total Preds: {total_preds}\n')

    if FULL_STATS: print("Scores:")
    hits, misses = regular_score(annotations, predictions)
    calculate_score(total_gold, total_preds, hits, misses)

    if FULL_STATS:
        print("\nGeneralized Scores:")
        hits, misses = generalized_score(annotations, hits, misses)
        calculate_score(total_gold, total_preds, hits, misses)


if __name__ == '__main__':
    eval_score(GOLD_FILE, PREDS_FILE, FULL_STATS=True)
