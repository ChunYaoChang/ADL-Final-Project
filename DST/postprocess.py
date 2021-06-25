import json
from collections import OrderedDict
from state_to_csv import write_csv 
import sys

output = {}
input_path = sys.argv[1]
output_path = sys.argv[2]
with open(args.input_path,'r') as f:
    prediction = json.load(f)
    for dial_id in prediction:
        turns = prediction[dial_id]["turns"]
        answer = {}
        for turn_id in range(len(turns)):
            turn = turns[str(turn_id)]
            for slot in turn["pred_belief"]:
                slot_name = "-".join(slot.split("-")[:-1])
                slot_value = slot.split("-")[-1]
                answer[slot_name] = slot_value
        output[dial_id] = answer


write_csv(output, output_path)



