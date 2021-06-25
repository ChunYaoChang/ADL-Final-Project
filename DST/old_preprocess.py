import json
from collections import OrderedDict
import re
import os
from tqdm import tqdm

modes = ['train', 'dev', 'test_seen', 'test_unseen']
num_files = [138, 20, 16, 5]
data_length = dict(zip(modes, num_files))
Multiwoz_domains= ['hotel','train','attraction','restaurant', 'hospital', 'taxi', 'bus', 'police']

data_path = './data'
output_path = './preprocessed_data'
ontology_path = './ontology'
slot_desc_path = './slot_description'

def preprocess_dialog(mode):
    dial_dstc, dial_woz, dial_all = [], [], []
    for index in tqdm(range(1, data_length[mode]+1)):
        with open(f'{data_path}/{mode}/dialogues_{index:03}.json', 'r') as f:
            data = json.load(f)
            for dialogue in data:
                dial_id = dialogue['dialogue_id']
                turn_round = len(dialogue['turns']) - 1
                
                domains = [domain.lower() for domain in dialogue['services']]
                turns = []
                turn_example = {'system': 'none', 'user': 'none', 'state': {'active_intent': 'none', 'slot_values': {}}}

                if 'test' not in mode:
                    for frame in dialogue['turns'][0]['frames']:
                        if 'state' in frame.keys():
                            for slot_name in frame['state']['slot_values']:
                                for slot_value in frame['state']['slot_values'][slot_name]:
                                    if(f'{frame['service'].lower()}_{slot_name}') not in turn_example['state']['slot_values']:
                                        turn_example['state']['slot_values'][f'{frame['service'].lower()}_{slot_name}'] = slot_value.lower()

                turn_example['user'] = dialogue['turns'][0]['utterance'].lower()
                turns.append(turn_example)

                for turn_idx in range(1, turn_round, 2):
                    turn_example = {'system' : 'none', 'user' : 'none', 'state' : {'active_intent' : 'none', 'slot_values' : {}}}
                    user_turn = dialogue['turns'][turn_idx + 1]
                    system_turn = dialogue['turns'][turn_idx]      
                    turn_example['user'] = user_turn['utterance'].lower()
                    turn_example['system'] = system_turn['utterance'].lower()
                    if 'test' not in mode:
                        for frame in dialogue['turns'][turn_idx + 1]['frames']:
                            if 'state' in frame.keys():
                                for slot_name in frame['state']['slot_values']:
                                    for slot_value in frame['state']['slot_values'][slot_name]:
                                        if(f'{frame['service'].lower()}_{slot_name}') not in turn_example['state']['slot_values']:
                                            turn_example['state']['slot_values'][f'{frame['service'].lower()}_{slot_name}'] = slot_value.lower()
                    turns.append(turn_example)

                if not re.match(r'[A-Z]+', dial_id):
                    dial_dstc.append({'dial_id' : dial_id, 'domains' : domains, 'turns' : turns})
                else:
                    dial_woz.append({'dial_id' : dial_id, 'domains' : domains, 'turns' : turns})
                dial_all.append({'dial_id' : dial_id, 'domains' : domains, 'turns' : turns})
    with open(f'{output_path}/{mode}_dstc.json', 'w') as f:
        json.dump(dial_dstc, f, indent=2)
    with open(f'{output_path}/{mode}_woz.json', 'w') as f:
        json.dump(dial_woz, f, indent=2)
    with open(f'{output_path}/{mode}_all.json', 'w') as f:
        json.dump(dial_all, f, indent=2)

def create_ontology():
    service_name_all = []
    ontology_dstc = OrderedDict()
    ontology_woz = OrderedDict()
    ontology_all = OrderedDict()
    with open(f'{data_path}/schema.json','r') as f:
        schema = json.load(f)
        for service in schema:
            for slot in service['slots']:
                if service['service_name'] not in Multiwoz_domains:
                    ontology_dstc[f'{service['service_name]}-{slot['name]}'] = []
                    if service['service_name'] not in service_name_all:
                        service_name_all.append(service['service_name'])
                else:
                    ontology_woz[f'{service['service_name]}-{slot['name]}'] = []
                ontology_all[f'{service['service_name]}-{slot['name]}'] = []

    for mode in modes[:2]:
        for index in tqdm(range(1, data_length[mode])):
            with open(f'{data_path}/{mode}/dialogues_{index:03}.json', 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    for turn in dialogue['turns']:
                        for frame in turn['frames']:
                            if 'state' not in frame.keys():
                                for slot_name in frame['state']['slot_values']:
                                    for slot_value in frame['state']['slot_values'][slot_name]:
                                        if frame['service'] not in Multiwoz_domains:
                                            if slot_value not in ontology_dstc[f'{frame['service']}-{slot_name}']:
                                                ontology_dstc[f'{frame['service']}-{slot_name}'] += [slot_value]
                                        else:
                                            if slot_value not in ontology_woz[f'{frame['service']}-{slot_name}']:
                                                ontology_woz[f'{frame['service']}-{slot_name}'] += [slot_value]
                                        if slot_value not in ontology_all[f'{frame['service']}-{slot_name}']:
                                                ontology_all[f'{frame['service']}-{slot_name}'] += [slot_value]

    with open(f'{ontology_path}/ontology_dstc.json', 'w') as f:
        json.dump(ontology_dstc, f, indent=2)
    with open(f'{ontology_path}/ontology_woz.json', 'w') as f:
        json.dump(ontology_woz, f, indent=2)
    with open(f'{ontology_path}/ontology_all.json', 'w') as f:
        json.dump(ontology_all, f, indent=2)

def slot_description():
    d = OrderedDict()
    d2 = OrderedDict()
    d3 = OrderedDict()
    d4 = OrderedDict()
    d5 = OrderedDict()
    with open(f'{data_path}/schema.json','r') as f:
        schema = json.load(f)
    for service in schema:
        for slot in service['slots']:
            #if True:
            if service['service_name'] not in Multiwoz_domains:
                d[(f'{service['service_name]}-{slot['name]}').lower()] = {
                        'description_human' : service['description'] + ', ' + slot['description'].lower() ,
                        'values' : slot['possible_values'] if 'possible_values' in slot.keys() else [],
                        'naive' : 'none',
                        'question' : 'none',
                        'slottype' : 'none'
                        }
                d3[(f'{service['service_name]}-{slot['name]}').lower()] = {
                        'description_human' : slot['description'].lower() ,
                        'values' : slot['possible_values'] if 'possible_values' in slot.keys() else [],
                        'naive' : 'none',
                        'question' : 'none',
                        'slottype' : 'none'
                        }
                d4[(f'{service['service_name]}-{slot['name]}').lower()] = {
                        'description_human' : service['description'] + ', ' + slot['description'].lower() ,
                        'values' : slot['possible_values'] if 'possible_values' in slot.keys() else [],
                        'naive' : 'none',
                        'question' : 'none',
                        'slottype' : 'none'
                        }

                               
            else:
                d2[(f'{service['service_name]}-{slot['name]}').lower()] = {
                        'description_human' : slot['description'].lower() ,
                        'values' : slot['possible_values'] if 'possible_values' in slot.keys() else [],
                        'naive' : 'none',
                        'question' : 'none',
                        'slottype' : 'none'
                        }
            
                d4[(f'{service['service_name]}-{slot['name]}').lower()] = {
                        'description_human' : slot['description'].lower() ,
                        'values' : slot['possible_values'] if 'possible_values' in slot.keys() else [],
                        'naive' : 'none',
                        'question' : 'none',
                        'slottype' : 'none'
                        }


            d5[(f'{service['service_name]}-{slot['name]}').lower()] = {
                        'description_human' : slot['description'].lower() ,
                        'values' : slot['possible_values'] if 'possible_values' in slot.keys() else [],
                        'naive' : 'none',
                        'question' : 'none',
                        'slottype' : 'none'
                        }
    
    with open(f'{slot_desc_path}/slot_description_dstc.json', 'w') as f:
        json.dump(d, f, indent=2)
    with open(f'{slot_desc_path}/slot_description_dstc_wo_domain_desc.json', 'w') as f:
        json.dump(d3, f, indent=2)
    with open(f'{slot_desc_path}/slot_description_woz.json', 'w') as f:
        json.dump(d2, f, indent=2)
    with open(f'{slot_desc_path}/slot_description_all.json', 'w') as f:
        json.dump(d4, f, indent=2)
    with open(f'{slot_desc_path}/slot_description_all_wo_domain_desc.json', 'w') as f:
        json.dump(d5, f, indent=2)


if __name__ == '__main__':
    for mode in modes:
        preprocess_dialog(mode)
    create_ontology()
    slot_description()
