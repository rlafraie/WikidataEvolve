#
# MIT License
#
# Copyright (c) 2020 Rashid Lafraie
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject_ect to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from pathlib import Path
import bz2
import datetime
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import shutil
from sklearn.model_selection import train_test_split
import operator
from pprint import pprint
from random import randrange, sample, seed
import numpy as np

# Random seed for testing
rand_seed_ = 28
random_state_ = rand_seed_  # for sklearn
seed(rand_seed_)  # random module
np.random.seed(rand_seed_)  # for numpy


def divide_triple_operation_stream(stream, num_intervals):
    # Divide stream of triple operations into <num_intervals> (nearly) equal-sized parts

    avg = len(stream) / float(num_intervals)
    out = []
    last = 0.0

    while last < len(stream):
        out.append(stream[int(last):int(last + avg)])
        last += avg

    return out


def write_to_triple_file(file, triples_iterable, entity_mapping_dict=None, relation_mapping_dict=None):
    # Store iterable of triples to file

    with file.open(mode="wt", encoding="UTF-8") as f:
        for triple in triples_iterable:
            subject_, object_, predicate_ = triple

            if entity_mapping_dict:
                subject_ = entity_mapping_dict[subject_]
                object_ = entity_mapping_dict[object_]
                predicate_ = relation_mapping_dict[predicate_]

            output_line = ' '.join((subject_, object_, predicate_)) + '\n'
            f.write(output_line)


def write_to_tc_file(file, triple_list, truth_value_list):
    # Store samples for triple classification to file

    with file.open(mode="wt", encoding="UTF-8") as f:
        for triple, truth_value in zip(triple_list, truth_value_list):
            subject_, object_, predicate_ = triple
            output_line = " ".join(map(str, (subject_, object_, predicate_, truth_value))) + '\n'
            f.write(output_line)


def write_to_mapping_files(output_path, element_dict, is_entity_dict):
    # Store entries of entity or relation mapping dict to file

    dict_file_name = "entity2id.txt" if is_entity_dict else "relation2id.txt"
    output_file = output_path / dict_file_name
    with output_file.open(mode="wt", encoding="UTF-8") as out:
        for wikidata_id, mapped_id in element_dict.items():
            out.write(" ".join(map(str, (mapped_id, wikidata_id))) + '\n')


def save_element_mapping_files(output_path, element_dict, is_entity_dict):
    # Stores id mapping of entities or relations to file.

    datasets_paths = [dir_ for dir_ in output_path.iterdir() if dir_.is_dir()]
    for path in datasets_paths:
        write_to_mapping_files(path, element_dict, is_entity_dict)


def map_triple_ids(triple_iterable, output_file, entities_dict, relations_dict, next_ent_id, next_rel_id,
                   contains_triple_op=True):
    # Map Wikidata identifiers to new, enumerated identifiers

    mapped_output_list = []

    for t in triple_iterable:
        if contains_triple_op:
            head, tail, rel, operations_type, ts = t
        else:
            head, tail, rel = t

        if head not in entities_dict:
            entities_dict[head] = next_ent_id
            next_ent_id += 1
        head = entities_dict[head]

        if tail not in entities_dict:
            entities_dict[tail] = next_ent_id
            next_ent_id += 1
        tail = entities_dict[tail]

        if rel not in relations_dict:
            relations_dict[rel] = next_rel_id
            next_rel_id += 1
        rel = relations_dict[rel]

        output = (head, tail, rel, operations_type, ts) if contains_triple_op else (head, tail, rel)
        mapped_output = tuple(map(str, output))
        mapped_output_list.append(mapped_output)

        output_file.write(" ".join(mapped_output) + "\n")

    return mapped_output_list, entities_dict, relations_dict, next_ent_id, next_rel_id


def map_triple_operations_identifiers(triple_operations_divided, output_path):
    # Iterate through triple operations and map wikidata_ids to new ids

    new_triple_operations_intervals = []
    static_triple_file = output_path / "mapped_triple-op2id.txt"

    entities_dict = {}
    relations_dict = {}
    next_ent_id = 0
    next_rel_id = 0

    with static_triple_file.open(mode="wt", encoding="UTF-8") as out:
        for triple_operations_stream in triple_operations_divided:
            mapping_output = map_triple_ids(triple_operations_stream, out, entities_dict, relations_dict, next_ent_id,
                                            next_rel_id)
            mapped_triple_operations_stream, entities_dict, relations_dict, next_ent_id, next_rel_id = mapping_output
            new_triple_operations_intervals.append(mapped_triple_operations_stream)

    return new_triple_operations_intervals, entities_dict, relations_dict


def create_global_mapping(triple_operations_divided, output_path):
    # Maps the Wikidata identifiers in the Wikidata9M dataset to enumerated identifiers and
    # stores the corresponding mapping dictionaries

    # (1) Map item and property ids of wikidata to new global entity and relation ids
    global_mapping_output = map_triple_operations_identifiers(triple_operations_divided, output_path)
    new_triple_operations_intervals, entities_dict, relations_dict = global_mapping_output
    print("Basic statistics: global number of entities: {}.".format(len(entities_dict)))
    print("Basic statistics: global number of relations: {}.".format(len(relations_dict)))

    # (2) Store entity2id and relation2id mapping
    save_element_mapping_files(output_path, entities_dict, is_entity_dict=True)
    save_element_mapping_files(output_path, relations_dict, is_entity_dict=False)

    return new_triple_operations_intervals


def get_triple_operations_stream(triple_operations_file):
    # Load sequence of triple operations from Wikidata9M into a list

    triple_operations_list = []
    with bz2.open(triple_operations_file, mode="rt", encoding="UTF-8") as f:
        for line in f:
            subject_ect_, object_, predicate_, operations_type, ts = line.split()
            triple_operations_list.append((int(subject_ect_), int(object_), int(predicate_), operations_type, ts))

    return triple_operations_list


def sort_triple_ops_list(triple_ops_list):
    # Sort list of triple operations with respect to their timestamps

    sorted_triple_operations = sorted(triple_ops_list, key=operator.itemgetter(4, 0, 1, 2, 3))

    return sorted_triple_operations


def create_path_structure(output_path, num_snaps):
    # Create path structure of WikidataEvolve

    subdir_names = ["snapshots", "updates", "increments"]
    paths_dict = {}

    for name in subdir_names:
        sub_dir_path = output_path / name
        sub_dir_path.mkdir(exist_ok=True)
        paths_dict[name] = sub_dir_path

        for snap in range(1, num_snaps + 1):
            snap_path = sub_dir_path / str(snap)
            snap_path.mkdir(exist_ok=True)

    return paths_dict


def copy_mapping_files_to_increments_path(paths, num_intervals):
    # Copy entity and relation mapping files to sub paths, i.e. increment paths

    increments_path = paths["increments"]
    entity_file = increments_path / "entity2id.txt"
    relation_file = increments_path / "relation2id.txt"

    for interval in range(1, num_intervals + 1):
        # Copy entity2id.txt and relation2id.txt to snapshot folders
        increment_fld = increments_path / "{}".format(interval)
        increment_entity_mapping_file = increment_fld / "entity2id.txt"
        increment_relation_mapping_file = increment_fld / "relation2id.txt"
        shutil.copy(str(entity_file), str(increment_entity_mapping_file))
        shutil.copy(str(relation_file), str(increment_relation_mapping_file))

        # Copy global_triple2id.txt to triple2id.txt for every snap
        global_triple_file = increment_fld / "global_triple2id.txt"
        triple_file = increment_fld / "triple2id.txt"
        shutil.copy(str(global_triple_file), str(triple_file))


def configure_training_increments(paths, num_intervals):
    # Compiles the training increments of the knowledge graph. An increment subsumes all triples which have been added
    # to the knowledge graph in between two snapshots (i.e. within an knowledge graph update). So the 
    # training increment is determined by comparing two consecutive training.

    snapshots_path = paths["snapshots"]
    increments_dataset_path = paths["increments"]

    previous_training_snapshot_triples = set()
    for interval in range(1, num_intervals + 1):
        current_training_snapshot_triples = load_snapshot_triple_set(snapshots_path, interval, filename="train2id.txt")
        training_increment = current_training_snapshot_triples - previous_training_snapshot_triples
        training_increment_file = increments_dataset_path / "{}".format(interval) / "train2id.txt"
        write_to_triple_file(training_increment_file, training_increment)
        print("Increment {}: number of training triples: {}.".format(interval, len(training_increment)))

        previous_training_snapshot_triples = current_training_snapshot_triples


def compile_knowledge_graph_updates(triple_operations_divided, updates_dataset_path):
    # Math triples in training snapshots to their corresponding triple operations to
    # compile a (training) update

    for interval_idx, triple_operations_list in enumerate(triple_operations_divided):
        output_lines = []
        for op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = op
            out_line = " ".join((subject_, object_, predicate_, operations_type)) + '\n'
            output_lines.append(out_line)

        interval = interval_idx + 1  # Because we count from snapshot 1
        update_file = updates_dataset_path / "{}".format(interval) / "triple-op2id.txt"
        with update_file.open(mode="wt", encoding="UTF-8") as output:
            output.writelines(output_lines)


def compile_knowledge_graph_snapshots(triple_operations_divided, paths_dict):
    # Determine snapshots from stream of triple operations which hold in the knowledge graph
    # at the intervals' ends

    triple_result_set = set()
    for interval_idx, triple_operations_list in enumerate(triple_operations_divided):
        for op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = op
            triple = (subject_, object_, predicate_)

            if operations_type == "+":
                triple_result_set.add(triple)

            if operations_type == "-":
                triple_result_set.remove(triple)

        snapshot = interval_idx + 1

        # Create global_triple2id.txt for every dataset
        for dataset_path in paths_dict.values():
            global_triple2id = dataset_path / "{}".format(snapshot) / "global_triple2id.txt"
            write_to_triple_file(global_triple2id, triple_result_set)


def load_snapshot_triple_set(path, snapshot, filename="global_triple2id.txt"):
    # Load file of triples into a set

    triple_file = path / str(snapshot) / "{}".format(filename)
    triple_set = set()
    with triple_file.open(mode="rt", encoding="UTF-8") as f:
        for line in f:
            subject_, object_, predicate_ = line.split()
            triple_set.add((subject_, object_, predicate_))
    return triple_set


def detect_added_and_deleted_triples(paths, prev_snapshot, curr_snapshot):
    # Determine from two successive snapshots which triples have been added and deleted
    # in the intermediate interval

    snapshots_path = paths["snapshots"]
    old_triples_set = set() if prev_snapshot == 0 else load_snapshot_triple_set(snapshots_path, prev_snapshot)
    new_triples_set = load_snapshot_triple_set(snapshots_path, curr_snapshot)

    added_triples_set = new_triples_set - old_triples_set
    deleted_triples_set = old_triples_set - new_triples_set

    return added_triples_set, deleted_triples_set


def determine_reinserted_triples(added_triples_set, triple_sets, triple_histories):
    # Determine triples which have been inserted into the knowledge graph again after their deletion

    reinserted_triples_set = set()
    for triple in added_triples_set:
        # Reinserts for train triples
        if triple in triple_histories["train_triples_history"]:
            # Attach to train set to exclude it from split
            triple_sets["train_triples_set"].add(triple)
            reinserted_triples_set.add(triple)

        # Reinserts for valid triples
        if triple in triple_histories["valid_triples_history"]:
            # Attach to train set to exclude it from split
            triple_sets["valid_triples_set"].add(triple)
            reinserted_triples_set.add(triple)

        # Reinserts for test triples
        if triple in triple_histories["test_triples_history"]:
            # Attach to train set to exclude it from split
            triple_sets["test_triples_set"].add(triple)
            reinserted_triples_set.add(triple)

    return reinserted_triples_set


def perform_train_valid_test_split(added_triples_set):
    # Train- / valid- / test- split on newly added triples

    data = list(added_triples_set)
    # Split triple result list to 90% train and 10% eval triple
    train_triples, eval_triples = train_test_split(data, test_size=0.1, random_state=random_state_)
    valid_triples, test_triples = train_test_split(eval_triples, test_size=0.5, random_state=random_state_)

    return train_triples, valid_triples, test_triples


def update_triple_records(triple_sets, triple_histories, train_triples, valid_triples, test_triples):
    # Update sets used to track currently contained triples and triples which once emerged
    # in the knowledge graph

    triple_sets["train_triples_set"].update(train_triples)
    triple_histories["train_triples_history"].update(train_triples)

    triple_sets["valid_triples_set"].update(valid_triples)
    triple_histories["valid_triples_history"].update(valid_triples)

    triple_sets["test_triples_set"].update(test_triples)
    triple_histories["test_triples_history"].update(test_triples)


def remove_deleted_triples(deleted_triples_set, triple_sets):
    # Remove deleted triples from sets to track triples currently contained in the knowledge graph

    for triple in deleted_triples_set:
        if triple in triple_sets["train_triples_set"]:
            triple_sets["train_triples_set"].remove(triple)

        if triple in triple_sets["valid_triples_set"]:
            triple_sets["valid_triples_set"].remove(triple)

        if triple in triple_sets["test_triples_set"]:
            triple_sets["test_triples_set"].remove(triple)


def save_current_records_to_file(paths, triple_sets, interval):
    # Save currently contained triples to sub folders of current interval

    # Static train
    static_train_file = paths["snapshots"] / "{}".format(interval) / "train2id.txt"
    write_to_triple_file(static_train_file, triple_sets["train_triples_set"])

    # Test (all)
    snapshot_test_file = paths["snapshots"] / "{}".format(interval) / "test2id.txt"
    write_to_triple_file(snapshot_test_file, triple_sets["test_triples_set"])

    update_test_file = paths["updates"] / "{}".format(interval) / "test2id.txt"
    write_to_triple_file(update_test_file, triple_sets["test_triples_set"])

    increment_test_file = paths["increments"] / "{}".format(interval) / "test2id.txt"
    write_to_triple_file(increment_test_file, triple_sets["test_triples_set"])

    # Valid (all)
    snapshot_valid_file = paths["snapshots"] / "{}".format(interval) / "valid2id.txt"
    write_to_triple_file(snapshot_valid_file, triple_sets["valid_triples_set"])

    update_valid_file = paths["updates"] / "{}".format(interval) / "valid2id.txt"
    write_to_triple_file(update_valid_file, triple_sets["valid_triples_set"])

    increment_valid_file = paths["increments"] / "{}".format(interval) / "valid2id.txt"
    write_to_triple_file(increment_valid_file, triple_sets["valid_triples_set"])

    print("Snapshot {}: number of validation triples: {}.".format(interval, len(triple_sets["valid_triples_set"])))
    print("Snapshot {}: number of test triples: {}.".format(interval, len(triple_sets["test_triples_set"])))


def map_global_to_snapshot_identifiers(paths, num_snapshots):
    # Traverse all snapshots and create local identifier mappings

    for snapshot in range(1, num_snapshots + 1):
        create_snapshot_entity_and_relation_mapping(paths, snapshot)


def create_snapshot_entity_and_relation_mapping(paths, snapshot):
    # Map global identifiers to local identifiers which hold within each snapshot

    static_dataset_path = paths["snapshots"]
    snapshot_fld = static_dataset_path / str(snapshot)
    triple_set_filenames = ["train2id.txt", "valid2id.txt", "test2id.txt"]

    # Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    next_ent_id = 0
    next_rel_id = 0
    entities_dict = {}
    relations_dict = {}

    # Iterate through triple operations and map wikidata_ids to new ids
    for dataset in triple_set_filenames:
        triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, dataset)
        print("Snapshot {}: number of {} triples (static dataset): {}.".format(snapshot, dataset[:dataset.find("2")],
                                                                               len(triple_set)))

        static_triple_file = snapshot_fld / "{}".format(dataset)
        with static_triple_file.open(mode="wt", encoding="UTF-8") as out:
            mapping_output = map_triple_ids(triple_set, out, entities_dict, relations_dict, next_ent_id,
                                            next_rel_id, contains_triple_op=False)
            mapped_triple_set, entities_dict, relations_dict, next_ent_id, next_rel_id = mapping_output

    # Store entity2id and relation2id mapping
    print("Snapshot {}: number of entities: {}.".format(snapshot, len(entities_dict)))
    write_to_mapping_files(snapshot_fld, entities_dict, is_entity_dict=True)

    print("Snapshot {}: number of relations: {}.".format(snapshot, len(relations_dict)))
    write_to_mapping_files(snapshot_fld, relations_dict, is_entity_dict=False)


def configure_training_and_test_snapshots(paths, num_snapshots):
    # Traverse the Wikidata9M time stream and compile knowledge graph (training) snapshots and
    # evaluation datasets

    # Sets to record training, validation and test triples along the evolution of knowledge graph
    triple_sets = {"train_triples_set": set(),
                   "valid_triples_set": set(),
                   "test_triples_set": set()}

    # Sets to track associations between triples and the training and evaluation
    # datasets along the evolution of the knowledge graph
    triple_histories = {"train_triples_history": set(),
                        "valid_triples_history": set(),
                        "test_triples_history": set()}

    for snapshot in range(1, num_snapshots + 1):
        added_triples_set, deleted_triples_set = detect_added_and_deleted_triples(paths, snapshot - 1, snapshot)
        reinserted_triple_set = determine_reinserted_triples(added_triples_set, triple_sets, triple_histories)
        newly_added_triples_set = added_triples_set - reinserted_triple_set
        train_triples, valid_triples, test_triples = perform_train_valid_test_split(newly_added_triples_set)
        update_triple_records(triple_sets, triple_histories, train_triples, valid_triples, test_triples)
        remove_deleted_triples(deleted_triples_set, triple_sets)
        save_current_records_to_file(paths, triple_sets, snapshot)


def save_negative_triple_classification_examples(deleted_triples_set, positive_oscillated_triples_set,
                                                 negative_oscillated_triples_set, paths, dataset, snapshot):
    # Save test examples of categories of Negative Triple Classification to files.

    snapshot_folder = paths["updates"] / "{}".format(snapshot)

    deleted_triple_file = snapshot_folder / "tc_negative_deleted_{}_triples.txt".format(dataset)
    positive_oscillated_file = snapshot_folder / "tc_positive_oscillated_{}_triples.txt".format(dataset)
    negative_oscillated_file = snapshot_folder / "tc_negative_oscillated_{}_triples.txt".format(dataset)

    print("Interval {}.".format(snapshot))
    print("-- Deleted {} triples: {}.".format(dataset, len(deleted_triples_set)))
    print("-- Positive oscillating {} triples: {}.".format(dataset, len(positive_oscillated_triples_set)))
    print("-- Negative oscillating {} triples: {}.\n".format(dataset, len(negative_oscillated_triples_set)))

    if len(deleted_triples_set) > 0:
        write_to_tc_file(file=deleted_triple_file, triple_list=deleted_triples_set,
                         truth_value_list=[0] * len(deleted_triples_set))

    if len(negative_oscillated_triples_set) > 0:
        write_to_tc_file(file=negative_oscillated_file, triple_list=negative_oscillated_triples_set,
                         truth_value_list=[0] * len(negative_oscillated_triples_set))

    if len(positive_oscillated_triples_set) > 0:
        write_to_tc_file(file=positive_oscillated_file, triple_list=positive_oscillated_triples_set,
                         truth_value_list=[1] * len(positive_oscillated_triples_set))


def compile_deleted_and_oscillating_triples(paths, num_intervals, dataset="train"):
    # Traverse different intervals of Wikidata9M to track the current history of all triples
    # in order to attach them to the corresponding categories of Negative Triple Classification.
    # For a detailed description of the task see my master' thesis.

    # Transitions between categories:
    # (0)->(1) Inserted
    # (1)->(2) Deleted
    # (2)->(3) Positive Oscillated
    # (3)->(4) Negative Oscillated
    # (4)->(3) Positive Oscillated

    snapshots_path = paths["snapshots"]

    current_status_per_triple = {}
    deleted_triples_set = set()
    negative_oscillated_triples_set = set()
    positive_oscillated_triples_set = set()

    previous_dataset_triples = set()
    for interval in range(1, num_intervals + 1):
        # (8.1) Load [ train | valid | test ] triples from snapshot <snapshot_idx>
        current_dataset_triples = load_snapshot_triple_set(snapshots_path, interval,
                                                           filename="{}2id.txt".format(dataset))

        # Determine triples which have been inserted and deleted
        inserts = current_dataset_triples - previous_dataset_triples
        deletes = previous_dataset_triples - current_dataset_triples

        # Detect Deletes and Oscillating triples
        for triple in inserts:
            if triple in current_status_per_triple:
                status = current_status_per_triple[triple]

                if status == "Deleted":
                    status = "Positive Oscillated"
                    deleted_triples_set.remove(triple)
                    positive_oscillated_triples_set.add(triple)

                if status == "Negative Oscillated":
                    status = "Positive Oscillated"
                    negative_oscillated_triples_set.remove(triple)
                    positive_oscillated_triples_set.add(triple)

                current_status_per_triple[triple] = status
            else:
                current_status_per_triple[triple] = "Inserted"

        for triple in deletes:
            status = current_status_per_triple[triple]

            if status == "Inserted":
                status = "Deleted"
                deleted_triples_set.add(triple)

            if status == "Positive Oscillated":
                status = "Negative Oscillated"
                positive_oscillated_triples_set.remove(triple)
                negative_oscillated_triples_set.add(triple)

            current_status_per_triple[triple] = status

        previous_dataset_triples = current_dataset_triples

        save_negative_triple_classification_examples(deleted_triples_set, positive_oscillated_triples_set,
                                                     negative_oscillated_triples_set, paths, dataset, interval)


def compile_negative_triple_classification_test_samples(sub_paths_dict, num_intervals):
    compile_deleted_and_oscillating_triples(sub_paths_dict, num_intervals, "train")
    compile_deleted_and_oscillating_triples(sub_paths_dict, num_intervals, "valid")
    compile_deleted_and_oscillating_triples(sub_paths_dict, num_intervals, "test")


def verify_consistency(triple_ops_list, snapshot):
    triple_ops_dict = defaultdict(list)
    triple_dict = defaultdict(int)
    for ops in triple_ops_list:
        subject_, object_, predicate_, operations_type, ts = ops
        triple = (subject_, object_, predicate_)
        triple_dict[triple] = triple_dict[triple] + 1
        triple_ops_dict[triple].append((subject_, object_, predicate_, operations_type, ts))
        if triple_dict[triple] > 1:
            print("More than 1 operation for {} at snapshot {}.".format(triple, snapshot))
            pprint(triple_ops_dict[triple])


def determine_training_triple_operations(interval_triple_operations, inserted_train_triples, deleted_train_triples):
    # Match added and deleted triples of training dataset to their triple operations to compile
    # a training update.

    train_triple_operations = []
    assigned_triples = set()
    for operation in interval_triple_operations:
        subject_ect_, object_, predicate_, operations_type, ts = operation
        triple = (subject_ect_, object_, predicate_)

        if triple not in assigned_triples:
            if triple in inserted_train_triples and operations_type == "+":
                train_triple_operations.append((subject_ect_, object_, predicate_, operations_type, ts))
                assigned_triples.add(triple)
                if triple in deleted_train_triples and operations_type == "-":
                    train_triple_operations.append((subject_ect_, object_, predicate_, operations_type, ts))
                assigned_triples.add(triple)

                return train_triple_operations


def save_training_update(train_triple_operations, paths, interval):
    # Save training update to file.

    update_file = paths["updates"] / "{}".format(interval) / "train-op2id.txt"
    with update_file.open(mode="wt", encoding="UTF-8") as output:
        for triple_op in train_triple_operations:
            output_line = " ".join((triple_op[0], triple_op[1], triple_op[2], triple_op[3])) + '\n'
            output.write(output_line)


def configure_training_updates(triple_operations_divided, paths, num_intervals):
    old_train_triple_set = set()

    snapshots_path = paths["snapshots"]
    for interval in range(1, num_intervals + 1):
        # Load training snapshot
        new_train_triple_set = load_snapshot_triple_set(snapshots_path, interval, "train2id.txt")

        # Determine inserts and deletes
        inserted_train_triples = new_train_triple_set - old_train_triple_set
        deleted_train_triples = old_train_triple_set - new_train_triple_set

        # Match inserted and deleted triples to their corresponding triple ops
        interval_triple_operations = triple_operations_divided[interval - 1]
        train_triple_operations = determine_training_triple_operations(interval_triple_operations,
                                                                       inserted_train_triples, deleted_train_triples)

        # sort list of triple operations
        sorted(train_triple_operations, key=operator.itemgetter(4, 0, 1, 2, 3))
        verify_consistency(train_triple_operations, interval)

        # Save sorted training operations
        save_training_update(train_triple_operations, paths, interval)

        print("Update {}: number of train triple operations: {}.".format(interval, len(train_triple_operations)))
        print("-- number of insert operations: {}.".format(len(inserted_train_triples)))
        print("-- number of delete operations: {}.".format(len(deleted_train_triples)))

        old_train_triple_set = new_train_triple_set


def corrupt_triple(triple, mode, filter_set, entities_total):
    subject_, object_, predicate_ = triple
    while True:
        corr_entity = randrange(0, entities_total)
        negative_triple = (corr_entity, object_, predicate_) if mode == "head" else (subject_, corr_entity, predicate_)
        if negative_triple not in filter_set:
            break

    return negative_triple


def save_triple_classification_file(dataset_path, snapshot, positive_examples, negative_examples):
    output_file = dataset_path / str(snapshot) / "triple_classification_prepared_test_examples.txt"
    examples = positive_examples + negative_examples
    truth_values = [1] * len(positive_examples) + [0] * len(negative_examples)
    write_to_tc_file(output_file, examples, truth_values)


def compile_triple_classification_examples(paths_dict, num_intervals):
    # Compile text examples for the evaluation task of triple classification.
    # Exclude triples whose truth values oscillated throughout the Wikidata9M stream.

    updates_path = paths_dict["updates"]
    global_entities_file = updates_path / "entity2id.txt"
    entities_total = len(open(global_entities_file).readlines())
    print("-------------------------------------------------------------------------")
    print("Start gathering of persistent-negative examples for triple classification")
    print("Entities included in triple corruption process: {}.".format(entities_total))

    triple_set = set()
    test_triple_set = set()
    for interval in range(1, num_intervals + 1):
        # Load all triples which have ever been inserted
        snapshot_triples = load_snapshot_triple_set(updates_path, interval, filename="global_triple2id.txt")
        triple_set.update(snapshot_triples)

        # Load all triples which have ever been inserted
        snapshot_test_triples = load_snapshot_triple_set(updates_path, interval, filename="test2id.txt")
        test_triple_set.update(snapshot_test_triples)

    # Create pool of negative examples
    negative_triple_dict = {}  # test_triple -> negative example
    for triple in tqdm(test_triple_set):
        random_num = randrange(0, 10000)
        negative_triple = None

        # Corrupt head
        if random_num < 5000:
            negative_triple = corrupt_triple(triple, "head", triple_set, entities_total)

        # Corrupt tail
        elif random_num >= 5000:
            negative_triple = corrupt_triple(triple, "tail", triple_set, entities_total)

        negative_triple_dict[triple] = negative_triple

    # Iterate through snapshot to create triple_classification_file.txt from test2id.txt
    # files by adding negative examples
    for interval in range(1, num_intervals + 1):
        test_triples = list(load_snapshot_triple_set(updates_path, interval, filename="test2id.txt"))

        # Gather negative examples
        negative_examples = []
        for triple in test_triples:
            negative_triple = negative_triple_dict[triple]
            negative_examples.append(negative_triple)

        save_triple_classification_file(updates_path, interval, test_triples, negative_examples)


def sample_examples(paths_dict, interval, filename, num_samples):
    # Sample a fixed number of test examples from the compiled evaluation dataset of a
    # specific interval

    # Datasets to sample eval data for
    updates_path = paths_dict["updates"]
    increments_path = paths_dict["increments"]
    dataset_paths = [updates_path, increments_path]

    triple_set = load_snapshot_triple_set(updates_path, interval, filename)
    triple_set_sample = sample(triple_set, num_samples)

    for dataset_path in dataset_paths:
        # Rename file containing all triples to all_<filename>.txt (valid2id_all.txt | test2id_all.txt)
        snapshot_fld = dataset_path / "{}".format(interval)
        input_file = snapshot_fld / filename

        new_file_name = filename[:filename.find(".")] + "_all" + filename[filename.find("."):]
        input_file.rename(snapshot_fld / new_file_name)

        # Sample examples into new file with old name <filename>.txt (valid2id.txt | test2id.txt)
        sample_file = snapshot_fld / filename
        write_to_triple_file(sample_file, triple_set_sample)


def sample_evaluation_examples(paths_dict, num_intervals, num_samples):
    # Sample a fixed number of test examples from the compiled evaluation datasets to enable
    # a faster evaluation procedure.

    for interval in range(1, num_intervals + 1):
        # sample test and valid data for all datasets
        sample_examples(paths_dict, interval, "valid2id.txt", num_samples)
        sample_examples(paths_dict, interval, "test2id.txt", num_samples)


def remove_obsolet_triple_ops(triple_operations_divided):
    # Detect triples with an even number of triple operations in an interval
    # to delete their corresponding triple operations. This is done to reduce
    # the number of triple operations as an even number does not affect
    # the truth value of a triple between two snapshots.

    removed_triples = []
    new_snapshots_triple_operations = []
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        # To track operations for a triple in the interval before a snapshot
        triple_operation_dict = defaultdict(list)
        for triple_op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = triple_op

            if subject_ == object_:
                continue

            triple = (subject_, object_, predicate_)
            triple_operation_dict[triple].append(triple_op)

        # Determine all triples with a odd number of triple operations because
        # those with even numbers are inserted and deleted within the same interval
        filtered_triple_operations = []
        for triple, triple_op_list in triple_operation_dict.items():
            if len(triple_op_list) % 2 != 0:
                filtered_triple_operations.append(triple_op_list[-1])
            else:
                removed_triples.append(triple)

        sorted(filtered_triple_operations, key=operator.itemgetter(4, 0, 1, 2, 3))
        new_snapshots_triple_operations.append(filtered_triple_operations)

    return new_snapshots_triple_operations


def remove_uncommon_triple_ops(triple_operations_divided, num_snapshots, entity_frequencies_threshold,
                               relation_frequencies_threshold):
    # Filtering the Wikidata9M time stream of triples whose entities and relations have
    # a minimum frequency at the snapshots of Wikidata9M after their first emergence in
    # the knowledge graph.

    triple_operations_divided = remove_uncommon_entities(triple_operations_divided, num_snapshots,
                                                         entity_frequencies_threshold)
    triple_operations_divided = remove_uncommon_relations(triple_operations_divided, num_snapshots,
                                                          relation_frequencies_threshold)

    return triple_operations_divided


def remove_uncommon_relations(triple_operations_divided, num_snapshots, relation_frequencies_threshold):
    relation_occ_counter = defaultdict(lambda: {i: 0 for i in range(1, num_snapshots + 1)})

    triple_result_set = set()
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        for triple_op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = triple_op
            triple = (subject_, object_, predicate_)
            if operations_type == "+":
                triple_result_set.add(triple)
            elif operations_type == "-":
                triple_result_set.remove(triple)

        # Iterate through triple result list to obtain frequencies
        for triple in triple_result_set:
            subject_, object_, predicate_ = triple
            relation_occ_counter[predicate_][snapshot_idx + 1] += 1

    uncommon_relations = set()
    for relation, snapshot_frequencies_dict in relation_occ_counter.items():
        for snapshot, count in snapshot_frequencies_dict.items():
            if 0 < count < relation_frequencies_threshold:
                uncommon_relations.add(relation)
                break

    filtered_triple_operations_divided = []
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        filtered_ops = []
        for triple_op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = triple_op

            if predicate_ not in uncommon_relations:
                filtered_ops.append(triple_op)

        filtered_triple_operations_divided.append(filtered_ops)

    return filtered_triple_operations_divided


def remove_uncommon_entities(triple_operations_divided, num_snapshots, entity_frequencies_threshold):
    entity_occ_counter = defaultdict(lambda: {i: 0 for i in range(1, num_snapshots + 1)})

    triple_result_set = set()
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        for triple_op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = triple_op
            triple = (subject_, object_, predicate_)
            if operations_type == "+":
                triple_result_set.add(triple)
            elif operations_type == "-":
                triple_result_set.remove(triple)

        # Iterate through triple result list to obtain frequencies
        for triple in triple_result_set:
            subject_, object_, predicate_ = triple
            entity_occ_counter[subject_][snapshot_idx + 1] += 1
            entity_occ_counter[object_][snapshot_idx + 1] += 1

    uncommon_entities = set()
    for entity, snapshot_frequencies_dict in entity_occ_counter.items():
        for snapshot, count in snapshot_frequencies_dict.items():
            if 0 < count < entity_frequencies_threshold:
                uncommon_entities.add(entity)
                break

    filtered_triple_operations_divided = []
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        filtered_ops = []
        for triple_op in triple_operations_list:
            subject_, object_, predicate_, operations_type, ts = triple_op

            if subject_ not in uncommon_entities and object_ not in uncommon_entities:
                filtered_ops.append(triple_op)

        filtered_triple_operations_divided.append(filtered_ops)

    return filtered_triple_operations_divided


def load_mapping_dict(filepath):
    # Load mapping file into dict

    mapping_dict = {}
    with filepath.open(mode="rt", encoding="UTF-8") as f:
        for line in f:
            local_id, global_id = line.split()
            mapping_dict[global_id] = local_id

    return mapping_dict


def map_snapshot_datasets_to_local_identifiers(path_dict, num_intervals):
    # Map datasets of snapshots to enumerated, snapshot-bound identifiers.

    static_dataset_path = path_dict["snapshots"]
    dataset_name = "global_triple2id.txt"

    for interval in range(1, num_intervals + 1):
        snapshot_fld = static_dataset_path / "{}".format(interval)

        entities_mapping_file = snapshot_fld / "entity2id.txt"
        entity_mapping_dict = load_mapping_dict(entities_mapping_file)
        relations_mapping_file = snapshot_fld / "relation2id.txt"
        relation_mapping_dict = load_mapping_dict(relations_mapping_file)

        output_file_name = dataset_name[dataset_name.find("triple"):]
        output_file = snapshot_fld / output_file_name
        input_triple_set = load_snapshot_triple_set(static_dataset_path, interval, dataset_name)
        write_to_triple_file(output_file, input_triple_set, entity_mapping_dict, relation_mapping_dict)


def map_evaluation_samples_of_snapshots_to_local_ids(path_dict, num_intervals):
    # Map samples of evaluation datasets at the snapshot folders to local identifiers.

    updates_path = path_dict["updates"]
    static_dataset_path = path_dict["snapshots"]
    dataset_names = ["valid2id.txt", "test2id.txt"]

    for interval in range(1, num_intervals + 1):
        snapshots_folder = static_dataset_path / "{}".format(interval)

        entities_mapping_file = snapshots_folder / "entity2id.txt"
        entity_mapping_dict = load_mapping_dict(entities_mapping_file)

        relations_mapping_file = snapshots_folder / "relation2id.txt"
        relation_mapping_dict = load_mapping_dict(relations_mapping_file)

        for dataset in dataset_names:
            output_file_name = "sample_{}".format(dataset)
            output_file = snapshots_folder / output_file_name

            triple_set = load_snapshot_triple_set(updates_path, interval, dataset)
            write_to_triple_file(output_file, triple_set, entity_mapping_dict, relation_mapping_dict)

            # Switch sample and file with all valid | test examples
            all_triples_file = snapshots_folder / dataset
            new_file_name = dataset[:dataset.find(".")] + "_all" + dataset[dataset.find("."):]

            all_triples_file.rename(snapshots_folder / new_file_name)
            output_file.rename(snapshots_folder / dataset)


def filter_triple_operations(triple_operations_stream, num_intervals):
    # Filter triple operations to ensure each entity and relation has a minimum frequency at each snapshot

    triple_operations_stream = remove_obsolet_triple_ops(triple_operations_stream)
    triple_operations_stream = remove_uncommon_triple_ops(triple_operations_stream,
                                                          num_intervals,
                                                          entity_frequencies_threshold=15,
                                                          relation_frequencies_threshold=70)

    return triple_operations_stream


def compile_wikidataEvolve(triple_operations, num_intervals, filter_triple_ops=True, max_test_samples=None):
    current_ts = datetime.now().strftime("%H:%M:%S")
    print("Begin compilation of WikidataEvolve at {}.".format(current_ts))

    test_folder = "_4"

    output_folder = 'filtered_version_' if filter_triple_ops else 'raw_version_'
    output_folder += datetime.now().strftime("%Y%m%d") + test_folder

    output_path = Path.cwd() / 'datasets' / 'WikidataEvolve' / output_folder
    output_path.mkdir(exist_ok=True)

    # (1) Create separate directories for knowledge graph snapshots, updates and increments
    sub_paths_dict = create_path_structure(output_path, num_intervals)

    # (2) Split stream of triple_operations into <num_intervals> parts
    triple_operations_intervals = divide_triple_operation_stream(triple_operations, num_intervals)

    # (3) Filter triple operations
    if filter_triple_ops:
        triple_operations_intervals = filter_triple_operations(triple_operations_intervals, num_intervals)

    # (4) Mapping of Wikidata items' and properties' identifiers to enumerated identifiers
    triple_operations_intervals = create_global_mapping(triple_operations_intervals, output_path)

    # (5) Store each triple operations of each interval to corresponding update sub paths
    compile_knowledge_graph_updates(triple_operations_intervals, sub_paths_dict["updates"])

    # (6) Create snapshots ::== triples which hold after each knowledge graph update
    compile_knowledge_graph_snapshots(triple_operations_intervals, sub_paths_dict)

    # (6) Compile Training data for each snapshot and evaluation data for each knowledge graph
    #     snapshot, increment and update
    configure_training_and_test_snapshots(sub_paths_dict, num_intervals)

    # (7) Compile training increments
    configure_training_increments(sub_paths_dict, num_intervals)
    copy_mapping_files_to_increments_path(sub_paths_dict, num_intervals)

    # (7) Compile training updates
    configure_training_updates(triple_operations_intervals, sub_paths_dict, num_intervals)

    # (8) Compile test examples for categories of Negative Triple Classification
    compile_negative_triple_classification_test_samples(sub_paths_dict, num_intervals)

    # (9) Sample valid/ test examples if preferred
    if max_test_samples:
        sample_evaluation_examples(sub_paths_dict, num_intervals, max_test_samples)

    # (10) Assign global identifiers to enumerated, local identifiers which apply
    #      within a knowledge graph snapshot
    map_global_to_snapshot_identifiers(sub_paths_dict, num_intervals)

    if max_test_samples:
        map_evaluation_samples_of_snapshots_to_local_ids(sub_paths_dict, num_intervals)
    map_snapshot_datasets_to_local_identifiers(sub_paths_dict, num_intervals)

    # (11) Compile test examples for triple classification
    compile_triple_classification_examples(sub_paths_dict, num_intervals)
    print("Finished compilation process at {}.".format(datetime.now().strftime("%H:%M:%S")))


def main():
    triple_operation_file = Path.cwd() / "datasets" / "Wikidata9M.txt.bz2"
    triple_operations_stream = get_triple_operations_stream(triple_operation_file)

    number_of_intervals = int(input('In how many intervals do you want to divide the evolving knowledge graph?'))
    filter_input = input('Do you want to filter the graph\'s entities and relations - (y/n)?')
    filter_triple_operations_stream = True if filter_input.lower() == 'y' else False
    max_test_samples_input = input('Maximum number of test examples (For no cap leave blank and type enter): ')
    max_test_samples = int(max_test_samples_input) if max_test_samples_input.isnumeric() else None

    compile_wikidataEvolve(triple_operations_stream, number_of_intervals, filter_triple_operations_stream,
                           max_test_samples=max_test_samples)


if __name__ == '__main__':
    main()
