import random
import itertools
import time
from collections import defaultdict
import torch
from torch import nn
import datetime
from transformers import BertModel, BertTokenizer, AdamW, BertConfig
from difflib import SequenceMatcher
from flair.data import Sentence
from flair.models import SequenceTagger
from torch.nn import functional as F
import os
import numpy as np

# Set the seed value all over the place to make this reproducible.
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

MODEL_FILE = "./model_file"  # The path to the saved model
WORK_FOR = "Work_For"  # The relation of interest

TRAIN_INPUT_FILE = "./Corpus.TRAIN.txt"
TRAIN_GOLD_FILE = "./TRAIN.annotations"
DEV_INPUT_FILE = "./Corpus.DEV.txt"
DEV_GOLD_FILE = "./DEV.annotations"

# The prediction files for evaluation
TRAIN_PREDS_FILE = "./TRAIN.predictions"
DEV_PREDS_FILE = "./DEV.predictions"

# Special tokens for the target entities
PER_MARK_TOKEN = "[$]"
ORG_MARK_TOKEN = "[#]"

nlp = SequenceTagger.load('ner')


# Takes a time in seconds and returns a string hh:mm:ss.
def format_time(elapsed):

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class BertForRelationExtraction(nn.Module):

    def __init__(self, config, dropout=0.2):
        super(BertForRelationExtraction, self).__init__()

        # Load the pre-trained BERT model of 12-layers with an uncased vocab.
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=config)

        self.activation = nn.Tanh()  # Activation layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        self.entity_linear = nn.Sequential(
            self.dropout,
            self.activation,
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        )

        self.cls_linear = nn.Sequential(
            self.dropout,
            self.activation,
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        )

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(self.bert.config.hidden_size * 3, config.num_labels)
        )

    def forward(self, input, e1_span, e2_span):

        bert_output = self.bert(**input)
        output = bert_output[0]
        pooled_output = bert_output[1]

        # Pass the final hidden state of the ‘[CLS]’ token through activation layer and a fully connected layer.
        H0 = self.cls_linear(pooled_output)

        # Apply the average operation to get a vector representation for each of the two target entities.
        e1_avg = self.entity_average(output, e1_span)
        e2_avg = self.entity_average(output, e2_span)

        # Pass each of the two vectors through activation layer and a fully connected layer.
        H1 = self.entity_linear(e1_avg)
        H2 = self.entity_linear(e2_avg)

        # Concatenate the vectors and then pass the result through a fully connected layer.
        H = torch.cat([H0, H1.unsqueeze(0), H2.unsqueeze(0)], dim=1)
        out = self.classifier(H)

        return out

    def entity_average(self, hidden_output, entity_span):

        # Initialize the entity hidden vector.
        entity_hidden_vec = torch.zeros(entity_span[1] - entity_span[0] - 1, self.bert.config.hidden_size)

        # Add hidden state to entity_hidden_vec vector.
        for i, idx in enumerate(range(entity_span[0] + 1, entity_span[1])):
            entity_hidden_vec[i] = hidden_output[:, idx]

        # Apply average operation.
        average_tensor = torch.mean(entity_hidden_vec, dim=0)

        return average_tensor


class ProcessData:

    def __init__(self, data_path, gold_path=None):

        # Read the corpus file.
        self.raw_data = self.read_corpus_file(data_path)

        # Process the data
        self.processed_data = self.process_data()

        # Get the dataset ready.
        self.dataset = self.prepare_dataset()

        if gold_path is not None:  # No annotations file for the test set.
            self.gold = self.read_annotations_file(gold_path)

    @staticmethod
    def read_corpus_file(file_name):

        sentences = {}

        # Read the corpus file.
        with open(file_name, "r", encoding='utf-8') as f:
            lines = f.readlines()

        # For each line in the file
        for i, line in enumerate(lines):

            # If the file is in the wrong format raise ValueError.
            if i == 0 and line[0] == '#':
                raise ValueError("The input file is in the wrong format, it should be in the format of the .txt files.")

            sent_id, sent = line.strip().split("\t")
            sent = sent.replace("-LRB-", "(")
            sent = sent.replace("-RRB-", ")")
            sentences[sent_id] = sent

        return sentences

    def process_data(self):

        processed = []

        # For each sentence
        for sent_id, sent_str in self.raw_data.items():

            # Split the sentence into words by space.
            words = sent_str.split(" ")

            # Pass the pre-tokenized input to flair.
            sent = Sentence(words)
            nlp.predict(sent)

            # Build a dictionary that holds all the needed information about the sentence.
            sentence_dict = {'id': sent_id,
                             'sent': sent_str,
                             'entities': sent.get_spans('ner'),
                             'words': words}

            processed.append(sentence_dict)

        return processed

    @staticmethod
    def read_annotations_file(file_name):

        gold = []

        # Read the annotations file.
        with open(file_name, "r", encoding='utf-8') as f:
            annotations = f.readlines()

        # Save each annotation to the gold list
        for annotation in annotations:
            sent_id, subject, relation, object, sent = annotation.strip().split("\t")
            sent = sent.replace("-LRB-", "(")
            sent = sent.replace("-RRB-", ")")
            gold.append((sent_id, subject, relation, object, sent))

        return gold

    @staticmethod
    def get_entity_char_span(tokens, tokens_span):

        start_token, end_token = tokens_span

        # Get the position of the first character of the first token belongs to the entity.
        start_char = len(' '.join(word for word in tokens[:start_token + 1])) - len(tokens[start_token])
        # Get the position of the space after the last character of the last token belongs to the entity.
        end_char = len(' '.join(word for word in tokens[:end_token + 1]))

        return start_char, end_char

    def get_relevant_entities(self, sentence):

        # Extract the PER entities identified in the sentence together along with their location in the sentence.
        # --> For each PER entity convert its token span location into a chararcter span location.
        person_entities = [(entity.text,) + self.get_entity_char_span(sentence['words'], tuple(
            [int(token.split(" ")[1]) - 1 for token in [str(entity.tokens[0])] + [str(entity.tokens[-1])]])) for entity
                           in sentence['entities'] if entity.tag == 'PER']

        # Extract the ORG entities identified in the sentence together along with their location in the sentence.
        # --> For each ORG entity convert its token span location into a chararcter span location.
        organization_entities = [(entity.text,) + self.get_entity_char_span(sentence['words'], tuple(
            [int(token.split(" ")[1]) - 1 for token in [str(entity.tokens[0])] + [str(entity.tokens[-1])]])) for entity
                                 in sentence['entities'] if entity.tag == 'ORG']

        return person_entities, organization_entities

    def prepare_dataset(self):

        dataset = []

        for sentence in self.processed_data:

            # Find the PER and ORG entities in the sentence
            persons, organizations = self.get_relevant_entities(sentence)
            # Compute all permutations of (PER, ORG) entity pairs.
            permutations = [pair for pair in itertools.product(persons, organizations)]

            # Save the sentence's info with all possible pairs of entities (PER, ORG) in the sentence.
            dataset.append((sentence, permutations))

        return dataset


class RelationExtractionDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):  # Return the number of (PER, ORG) permutations.
        return sum([len(perms) for (_, perms) in self.dataset])


class TrainEvalModel:

    def __init__(self, model_state_dict=None):

        # Load the BERT tokenizer.
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # The number of output labels is 2 for binary classification.
        self.config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
        # Create an instance of BertForRelationExtraction model.
        self.model = BertForRelationExtraction(self.config)

        # Initialize the optimizer.
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

        # Define '[$]'and'[#]' as special tokens to the tokenizer and resize BERT's token embeddings table by 2 vectors.
        special_tokens_dict = {'additional_special_tokens': [PER_MARK_TOKEN, ORG_MARK_TOKEN]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.bert.resize_token_embeddings(len(self.tokenizer))

        # Binary Cross Entropy loss.
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Sigmoid layer for the evaluation.
        self.sigmoid = nn.Sigmoid()

        if model_state_dict is not None:  # Load the model state dict if supplied.
            self.model.load_state_dict(model_state_dict)

    def train(self, train_loader, train_gold, dev_loader, epochs):

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # I'll store a number of quantities such as training loss and training and dev scores.
        train_epochs_loss = []
        train_epochs_score = []
        dev_epochs_score = []

        max_f1 = 0.0  # Save the highest F1 score so far on the dev - initialized with 0.

        # For each epoch...
        for epoch in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print('\n=============================== Epoch {:} / {:} ===============================\n'.format(epoch + 1, epochs))
            print(' Training...\n')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Put the model into training mode.
            self.model.train()

            total_train_loss = 0.0  # Reset the total loss for this epoch.
            sum_loss = 0.0  # Reset the loss for the next 50 samples.

            # Tracking variables
            total = 0
            correct = 0

            # Save the predictions to file for later evaluation.
            with open(TRAIN_PREDS_FILE, "w") as f:

                # For each sentence and all of its (PER, ORG) entity pairs
                for (sentence, permutations) in train_loader:

                    # For each target entities pair - (PER, ORG)
                    for ((e1, e1_st, e1_end), (e2, e2_st, e2_end)) in permutations:

                        # Zero the gradients from the previous iteration.
                        self.optimizer.zero_grad()

                        # Get the input to the model and the (most likely) relation between the entities pair.
                        input, relation = self.prepare_train_sample(sentence, e1, e2, train_gold, (e1_st, e1_end),
                                                                    (e2_st, e2_end))

                        # Tokenize the sentence and add `[CLS]` and `[SEP]` tokens.
                        encoding = self.tokenizer(input, return_tensors="pt")

                        # Get the span of each target entity in the tokenized sentence,
                        e1_span = tuple([i for i, token in enumerate(encoding['input_ids'].tolist()[0]) if
                                         token == self.tokenizer.additional_special_tokens_ids[0]])
                        e2_span = tuple([i for i, token in enumerate(encoding['input_ids'].tolist()[0]) if
                                         token == self.tokenizer.additional_special_tokens_ids[1]])

                        # Perform a forward pass.
                        output = self.model(encoding, e1_span, e2_span)

                        # Create the label for this example.
                        is_work_for = 1 if relation == WORK_FOR else 0
                        gold = F.one_hot(torch.LongTensor([is_work_for]), 2).type_as(output)

                        # Compute the training loss.
                        loss = self.bce_loss(output, gold)

                        # Accumulate the training loss.
                        total_train_loss += loss.item()
                        sum_loss += loss.item()

                        # Perform a backward pass to calculate the gradients.
                        loss.backward()

                        # Update the parameters and take a step using the computed gradient.
                        self.optimizer.step()

                        with torch.no_grad():

                            # Get the model's prediction
                            prediction = torch.argmax(self.sigmoid(output))
                            total += 1

                            if prediction.item() == is_work_for:
                                correct += 1

                                # If the model identified the 'Work_For' relation, write this prediction to the file.
                                if prediction != 0:
                                    f.write(f"{sentence['id']}\t{e1}\t{WORK_FOR}\t{e2}\t( {sentence['sent']})\n")

                        # Progress update every 50 samples.
                        if total % 50 == 0:

                            # Compute the average loss of the last 50 samples.
                            current = sum_loss / 50

                            # Calculate elapsed time in minutes.
                            elapsed = format_time(time.time() - t0)

                            # Report progress.
                            print(f"   Samples: {total:>4,} / {len(train_loader):>4,}...",
                                  f" Elapsed: {elapsed}...",
                                  f" Train Loss: {current:.3f}...")

                            # Reset the loss for the next 50 samples.
                            sum_loss = 0.0

            # Calculate the average loss over all of the samples.
            avg_train_loss = total_train_loss / len(train_loader)
            train_epochs_loss.append(avg_train_loss)

            print()
            print("   Average training loss: {:.3f}".format(avg_train_loss))
            print("   Accuracy: {:.3f}".format(correct / len(train_loader)))
            print()

            # Compute and report precision, recall and F1 score fot the training predictions.
            f1, precision, recall = self.compute_score(TRAIN_GOLD_FILE, TRAIN_PREDS_FILE)
            train_epochs_score.append((f1, precision, recall))

            # Measure how long this training epoch took.
            training_time = format_time(time.time() - t0)

            print()
            print(f"   Training epoch took: {training_time}")

            # ========================================
            #               Evaluation
            # ========================================

            # After the completion of each training epoch, measure the model's performance on the dev set.
            print("\n\n Evaluating...\n")

            # Measure how long the evaluation takes.
            t1 = time.time()

            # Evaluate the model
            self.evaluate(dev_loader, DEV_PREDS_FILE, t1)
            print()

            # Compute and report precision, recall and F1 score for the dev predictions.
            f1, precision, recall = self.compute_score(DEV_GOLD_FILE, DEV_PREDS_FILE)
            dev_epochs_score.append((f1, precision, recall))

            if f1 > max_f1:

                # Save the current model's state.
                torch.save({'model_state_dict': self.model.state_dict()}, MODEL_FILE)
                print(f"\n   The current model's state saved to {MODEL_FILE}\n")
                max_f1 = f1

            # Measure how long the evaluation took.
            evaluation_time = format_time(time.time() - t1)
            print(f"   Evaluation took: {evaluation_time}\n\n")

            # Measure how long the whole epoch took - training and evaluating, together.
            epoch_time = format_time(time.time() - t0)
            print(f" Overall, Epoch took: {epoch_time}")

        print()
        print("Training completed!")
        print(f"Total training took {format_time(time.time() - total_t0)} (hh:mm:ss)\n")

        return train_epochs_loss, train_epochs_score, dev_epochs_score

    def prepare_train_sample(self, sentence, e1, e2, train_annotations, e1_span, e2_span):

        e1_start, e1_end = e1_span[0], e1_span[1]
        e2_start, e2_end = e2_span[0], e2_span[1]

        best_match_rate = (0, 0)
        best_annotation_match = None

        # For each annotation in the train annotations
        for (sent_id, subject, rel, object, _) in train_annotations:

            # Compute the match rate between e1 to subject and e2 to object.
            person_ent_match_rate = SequenceMatcher(None, subject, e1).ratio()
            organization_ent_match_rate = SequenceMatcher(None, object, e2).ratio()

            # If a match was found
            if sent_id == sentence['id'] and person_ent_match_rate >= 0.75 and organization_ent_match_rate >= 0.75:

                # Check if it's a better match than the best PER match rate and if so, update the variables.
                if person_ent_match_rate > best_match_rate[0]:
                    if organization_ent_match_rate >= best_match_rate[1]:
                        best_annotation_match = (sent_id, subject, rel, object)
                        best_match_rate = (person_ent_match_rate, organization_ent_match_rate)

                # Check if it's a better match than the best ORG match rate and if so, update the variables.
                elif organization_ent_match_rate > best_match_rate[1]:
                    if person_ent_match_rate >= best_match_rate[0]:
                        best_annotation_match = (sent_id, subject, rel, object)
                        best_match_rate = (person_ent_match_rate, organization_ent_match_rate)

        if best_annotation_match is not None:  # If a match was found

            _, gold_subject, gold_relation, gold_object = best_annotation_match
            relation = gold_relation  # Save the relation between the entities

            # Update e1 and e2 start and end positions.
            if gold_subject != e1:
                e1_start, e1_end = self.update_entity_span_bounds(e1, gold_subject, (e1_start, e1_end))
            if gold_object != e2:
                e2_start, e2_end = self.update_entity_span_bounds(e2, gold_object, (e2_start, e2_end))

        else:  # Otherwise, set the relation between the entity pair to be NIL (no relation).
            relation = 'NIL'

        # Mark the (updated) entities in the sentence.
        train_sample = self.mark_entities(sentence['sent'], (e1_start, e1_end), (e2_start, e2_end))

        return train_sample, relation

    @staticmethod
    def update_entity_span_bounds(entity, gold_entity, original_span):

        new_start = original_span[0]
        new_end = original_span[1]

        if len(entity) > len(gold_entity):
            if gold_entity.split(" ")[0] == entity.split(" ")[0]:
                new_end = original_span[0] + len(gold_entity)
            else:
                new_start += len(entity) - len(gold_entity)

        elif len(entity) < len(gold_entity):
            if gold_entity.split(" ")[0] == entity.split(" ")[0]:
                new_end = original_span[0] + len(gold_entity)
            else:
                new_start -= len(gold_entity) - len(entity)

        new_span = (new_start, new_end)
        return new_span

    @staticmethod
    def mark_entities(sent, per_entity_span, org_entity_span):

        e1_start, e1_end = per_entity_span[0], per_entity_span[1]
        e2_start, e2_end = org_entity_span[0], org_entity_span[1]

        if e1_end < e2_start:
            return sent[:e1_start] + f'{PER_MARK_TOKEN} ' + sent[e1_start:e1_end] + f' {PER_MARK_TOKEN}' + \
                sent[e1_end:e2_start] + f'{ORG_MARK_TOKEN} ' + sent[e2_start:e2_end] + f' {ORG_MARK_TOKEN}' + sent[e2_end:]

        return sent[:e2_start] + f'{ORG_MARK_TOKEN} ' + sent[e2_start:e2_end] + f' {ORG_MARK_TOKEN}' + \
            sent[e2_end:e1_start] + f'{PER_MARK_TOKEN} ' + sent[e1_start:e1_end] + f' {PER_MARK_TOKEN}' + sent[e1_end:]

    def evaluate(self, loader, out_file, time_stamp):

        # Put the model in evaluation mode.
        self.model.eval()

        # Tracking variables
        total_dev = 0
        work_for_dev = 0.0

        with torch.no_grad():

            # Save the predictions to file for later score computation.
            with open(out_file, "w") as f:

                # For each sentence and all of its (PER, ORG) entity pairs
                for (sentence, permutations) in loader:

                    # For each target entities pair - (PER, ORG)
                    for ((e1, e1_st, e1_end), (e2, e2_st, e2_end)) in permutations:

                        # Mark the target entities.
                        input = self.mark_entities(sentence['sent'], (e1_st, e1_end), (e2_st, e2_end))

                        # Tokenize the sentence and add `[CLS]` and `[SEP]` tokens.
                        encoding = self.tokenizer(input, return_tensors="pt")

                        # Get the span of each target entity in the tokenized sentence.
                        e1_span = tuple([i for i, token in enumerate(encoding['input_ids'].tolist()[0]) if
                                         token == self.tokenizer.additional_special_tokens_ids[0]])
                        e2_span = tuple([i for i, token in enumerate(encoding['input_ids'].tolist()[0]) if
                                         token == self.tokenizer.additional_special_tokens_ids[1]])

                        # Perform a forward pass.
                        output = self.model(encoding, e1_span, e2_span)

                        # Get the model's prediction.
                        prediction = torch.argmax(self.sigmoid(output))
                        total_dev += 1

                        # If the model identified the 'Work_For' relation, write this prediction to the file.
                        if prediction.item() != 0:
                            work_for_dev += 1
                            f.write(f"{sentence['id']}\t{e1}\t{WORK_FOR}\t{e2}\t( {sentence['sent']})\n")

                        # Progress update every 50 samples.
                        if total_dev % 50 == 0:

                            # Calculate elapsed time in minutes.
                            elapsed = format_time(time.time() - time_stamp)

                            # Report progress.
                            print(f"   Samples: {total_dev:>4,} / {len(loader):>4,}...",
                                  f" Elapsed: {elapsed}...",
                                  f" Work_For found: {work_for_dev}...")

    @staticmethod
    def extract_relations(path):

        id2relations = defaultdict(list)

        # Read the file.
        with open(path) as f:
            lines = f.readlines()

        for line in lines:  # For each annotation in the file
            sent_id, subject, relation, object, sentence = line.split("\t")

            if relation == WORK_FOR:  # Keep only the annotations that have a Work_For relation.
                id2relations[sent_id].append((subject, object, sentence))

        return id2relations

    def compute_score(self, gold_file, preds_file):

        # Get the relevant Work_For relations from both the annotations and predictions file.
        annotations = self.extract_relations(gold_file)
        predictions = self.extract_relations(preds_file)

        total_gold = sum([len(sent_annotations) for sent_annotations in annotations.values()])
        total_preds = sum([len(sent_annotations) for sent_annotations in predictions.values()])
        print(f'   Total Gold: {total_gold}')
        print(f'   Total Pred: {total_preds}')

        # Tracking variables
        hits = 0
        misses = []

        for sent_id, predicted_relations in predictions.items():
            for (per_pred, org_pred, _) in predicted_relations:

                found = False  # Indicator flag

                if sent_id in annotations:
                    sent_gold_relations = annotations[sent_id]

                    # For each annotation under the same sentence id as the prediction
                    for (per_gold, org_gold, sent) in sent_gold_relations:

                        # Search for a match while ignoring missing or extra determiners.
                        if per_gold == per_pred and (org_gold == org_pred or org_gold == 'the ' + org_pred or 'the ' +
                                                     org_gold == org_pred or org_gold == 'The ' + org_pred or 'The ' +
                                                     org_gold == org_pred):
                            found = True
                            hits += 1
                            # Remove this annotation to avoid from counting duplicates as hits.
                            annotations[sent_id].remove((per_gold, org_gold, sent))
                            break

                if not found:  # If the prediction didn't appear in the annotations.
                    misses.append((sent_id, per_pred, org_pred))

        # Calculate scores.
        precision = round((hits / total_preds) * 100, 2) if total_preds != 0 else 0
        recall = round((hits / total_gold) * 100, 2)
        f1 = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) != 0 else 0

        # Report scores.
        print(f'   Hit: {hits}')
        print(f'   Miss: {len(misses)}\n')
        print(f"   Precision: {precision}%")
        print(f"   Recall: {recall}%")
        print(f"   F1: {f1}%")

        return f1, precision, recall

    def predict(self, loader, output_file):

        # Put the model in evaluation mode.
        self.model.eval()

        with torch.no_grad():

            # Save the predictions to file.
            with open(output_file, "w") as out:

                # For each sentence and all of its (PER, ORG) entity pairs
                for (sentence, permutations) in loader:

                    # For each target entities pair - (PER, ORG)
                    for ((e1, e1_st, e1_end), (e2, e2_st, e2_end)) in permutations:

                        # Mark the target entities.
                        input = self.mark_entities(sentence['sent'], (e1_st, e1_end), (e2_st, e2_end))

                        # Tokenize the sentence and add `[CLS]` and `[SEP]` tokens.
                        encoding = self.tokenizer(input, return_tensors="pt")

                        # Get the span of each target entity in the tokenized sentence.
                        e1_span = tuple([i for i, token in enumerate(encoding['input_ids'].tolist()[0]) if
                                         token == self.tokenizer.additional_special_tokens_ids[0]])
                        e2_span = tuple([i for i, token in enumerate(encoding['input_ids'].tolist()[0]) if
                                         token == self.tokenizer.additional_special_tokens_ids[1]])

                        # Perform a forward pass.
                        output = self.model(encoding, e1_span, e2_span)
                        prediction = torch.argmax(self.sigmoid(output))

                        # If the model identified the 'Work_For' relation, write this prediction to the file.
                        if prediction.item() != 0:
                            out.write(f"{sentence['id']}\t{e1}\t{WORK_FOR}\t{e2}\t( {sentence['sent']})\n")
