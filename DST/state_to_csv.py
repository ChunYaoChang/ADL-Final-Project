"""
prepare your answers in the following format:
ans = {
    dialogue_id1: {service1-slot1: value1, service1-slot2: value2, ...},
    dialogue_id2: {service2-slot3: value3, service3-slot4: value4, ...},
}
output_path = 'submission.csv'
"""


def write_csv(ans, output_path):
    ans = sorted(ans.items(), key=lambda x: x[0])
    with open(output_path, 'w') as f:
        f.write('id,state\n')
        for dialogue_id, states in ans:
            if len(states) == 0:  # no state ?
                str_state = 'None'
            else:
                states = sorted(states.items(), key=lambda x: x[0])
                str_state = ''
                for slot, value in states:
                    # NOTE: slot = "{}-{}".format(service_name, slot_name)
                    str_state += "{}={}|".format(
                            slot.lower(), value.replace(',', '_').lower())
                str_state = str_state[:-1]
            f.write('{},{}\n'.format(dialogue_id, str_state))
