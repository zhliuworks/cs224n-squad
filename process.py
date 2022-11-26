import os
import numpy as np
from collections import defaultdict, OrderedDict
from tqdm import trange
from multiprocessing import Pool, cpu_count

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class SquadPreprocess:
    '''
    CS224n-specific SQuAD 2.0 dataset pre-processing
    '''
    def __init__(self, tokenizer, max_length=384, doc_stride=128):
        '''
        `max_length`: The maximum length of a feature (question and context)
        `doc_stride`: The authorized overlap between two part of the context when splitting it is needed
        '''
        self.tokenizer = tokenizer
        self.pad_on_right = (self.tokenizer.padding_side == 'right')
        self.max_length = max_length
        self.doc_stride = doc_stride

    def process_dataset(self, dataset, is_train):
        if is_train:
            return dataset.map(self._get_train_features, batched=True, remove_columns=dataset.column_names)
        else:
            dataset = dataset.map(self._get_test_features, batched=True, remove_columns=dataset.column_names)
            offset_mappings = dataset['offset_mapping']
            example_ids = dataset['example_id']
            dataset = dataset.remove_columns(['offset_mapping', 'example_id'])
            return dataset, offset_mappings, example_ids

    def _get_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples['question'] = [q.lstrip() for q in examples['question']]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples['question' if self.pad_on_right else 'context'],
            examples['context' if self.pad_on_right else 'question'],
            truncation='only_second' if self.pad_on_right else 'only_first',
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop('offset_mapping')

        # Let's label those examples!
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)

        return tokenized_examples

    def _get_test_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples['question'] = [q.lstrip() for q in examples['question']]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples['question' if self.pad_on_right else 'context'],
            examples['context' if self.pad_on_right else 'question'],
            truncation='only_second' if self.pad_on_right else 'only_first',
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

        # Let's label those examples!
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []

        for i, offsets in enumerate(tokenized_examples['offset_mapping']):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples['example_id'] = []

        for i in range(len(tokenized_examples['input_ids'])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples['example_id'].append(examples['id'][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples['offset_mapping'][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples['offset_mapping'][i])
            ]
        
        return tokenized_examples


class SquadPostprocess:
    '''
    Post-precess the model predictions (only when evaluation)
    '''
    def __init__(self, tokenizer, n_best_size=20, max_answer_length=30):
        '''
        `n_best_size`: The maximum number of possible answers considered
        `max_answer_length`: The maximum length of answers
        '''
        self.tokenizer = tokenizer
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

    def process_predictions(self, examples, features, offset_mappings, example_ids, all_start_logits, all_end_logits, batch_size=256):
        # Check whether the sizes are matched.
        num_examples = len(examples['id'])
        num_features = len(features['input_ids'])
        assert num_features == len(offset_mappings) == len(example_ids) == len(all_start_logits) == len(all_end_logits), \
            'The sizes of the inputs are not matched (except examples)'

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples['id'])}
        features_per_example = defaultdict(list)
        for i in range(num_features):
            features_per_example[example_id_to_index[example_ids[i]]].append(i)

        # The dictionaries we have to fill.
        predictions = OrderedDict()

        # Let's loop over all the examples!
        # Here we use multiprocessing to parallelize the process.
        pool = Pool(cpu_count())
        batch_args = []
        curr_index = 0
        for example_index in trange(num_examples, desc='Postprocess'):
            batch_args.append((
                features_per_example[example_index], examples['context'][example_index],
                offset_mappings, all_start_logits, all_end_logits, features
            ))
            if len(batch_args) >= batch_size:
                for answer in pool.starmap(self.process_one_example, batch_args):
                    predictions[examples['id'][curr_index]] = answer
                    curr_index += 1
                batch_args = []
        # the remaining examples
        for answer in pool.starmap(self.process_one_example, batch_args):
            predictions[examples['id'][curr_index]] = answer
            curr_index += 1
        pool.close()
                      
        return predictions

    def process_one_example(self, feature_indices, context, offset_mappings, all_start_logits, all_end_logits, features):
        min_null_score = None
        valid_answers = []
        
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = offset_mappings[feature_index]

            # Update minimum null prediction.
            cls_index = features['input_ids'][feature_index].index(self.tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -self.n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -self.n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            'score': start_logits[start_index] + end_logits[end_index],
                            'text': context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x['score'], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {'text': '', 'score': 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        answer = best_answer['text'] if best_answer['score'] > min_null_score else ''
        return answer


'''
Test functions
'''
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
from datasets import load_dataset

def init():
    print('# Loading datasets ...')
    data_files = {split: f'data/{split}-v2.0.hf.json' for split in ['train', 'dev', 'test']}
    datasets = load_dataset('json', data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    preprocessor = SquadPreprocess(tokenizer)
    return datasets, preprocessor, tokenizer

def test_process_train_set(datasets, preprocessor):
    print('# Test processing train set ...')
    train_dataset = preprocessor.process_dataset(datasets['train'], is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, collate_fn=default_data_collator)
    batch = next(iter(train_dataloader))
    print(batch.keys()) # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'])

def test_process_test_set(datasets, preprocessor):
    print('# Test processing test set ...')
    dev_dataset, offset_mappings, example_ids = preprocessor.process_dataset(datasets['dev'], is_train=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle=False, num_workers=8, collate_fn=default_data_collator)
    batch = next(iter(dev_dataloader))
    print(batch.keys()) # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'])
    return batch, datasets['dev'][:16], dev_dataset[:16], offset_mappings[:16], example_ids[:16]

def test_postprocess_predictions(batch, tokenizer, raw_dataset, pro_dataset, offset_mappings, example_ids):
    model = AutoModelForQuestionAnswering.from_pretrained('albert-base-v2')
    model.to('cuda:0')
    model.eval()
    postprocessor = SquadPostprocess(tokenizer)
    with torch.no_grad():
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        output = model(**batch)
        start_logits = output['start_logits'].tolist()
        end_logits = output['end_logits'].tolist()
        predictions = postprocessor.process_predictions(raw_dataset, pro_dataset, offset_mappings, example_ids, start_logits, end_logits)
    for id_, prediction in predictions.items():
        print(f'{id_}: {prediction}')

    
if __name__ == '__main__':
    datasets, preprocessor, tokenizer = init()
    test_process_train_set(datasets, preprocessor)
    batch, raw_dataset, pro_dataset, offset_mappings, example_ids = test_process_test_set(datasets, preprocessor)
    test_postprocess_predictions(batch, tokenizer, raw_dataset, pro_dataset, offset_mappings, example_ids)
