import json

def parse_schema():
    with open("./schema.json", "r") as f:
        schema = json.load(f)
    domain_to_slots = {}
    for service in schema:
        service_name = service["service_name"]
        slots = service["slots"]
        service_dict = {}
        for slot in slots:
            slot_name = slot["name"]
            try:
                service_dict[slot_name] = slot["possible_values"]
            except KeyError:
                service_dict[slot_name] = {}
        domain_to_slots[service_name] = service_dict
    return domain_to_slots

def add_tokens(tokenizer):
    # Adding special tokens to pretrained tokenizer
    # <st> : indicates system state
    # <ac> : indicates system action
    # <hi> : indicates dialogue history
    # <sys>: indicates turn of system
    # <usr>: indicates turn of user
    # <nm>: not mentioned
    with open("./schema.json", "r") as f:
        schema = json.load(f)
    additional_special_tokens = ["<st>", "<ac>", "<hi>", "<sys>", "<usr>", "<nm>"]
    for service in schema:
        service_name_token = "<" + service["service_name"] + ">"
        additional_special_tokens.append(service_name_token)
        slots = service["slots"]
        for slot in slots:
            slot_name_token = "<" + slot["name"] + ">"
            additional_special_tokens.append(slot_name_token)
            try:
                possible_values = slot["possible_values"]
                for possible_value in possible_values:
                    possible_value_name_token = "<" + possible_value + ">"
                    additional_special_tokens.append(possible_value_name_token)
            except KeyError:
                continue
    tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
