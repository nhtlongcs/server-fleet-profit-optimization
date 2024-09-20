import json
import uuid 
import threading
import pandas as pd
import math
from baseline.utils import load_problem_data
from animation import (
    rabbit_frames, finished_frame, clear
)
import time
import warnings
warnings.filterwarnings("ignore")

done_flag = threading.Event()

from baseline.evaluation_v6 import (
    get_capacity_by_server_generation_latency_sensitivity,
    check_datacenter_slots_size_constraint,
    put_fleet_on_hold,
    update_fleet
)
import sys
from tqdm import tqdm
if len(sys.argv) < 2:
    print('Usage: python cook_all.py <seed>')
    raise SystemExit
else:
    seed = sys.argv[1]

def df_to_submissions(all_action_df, path):
    solution = pd.DataFrame()
    
    _, datacenters, servers, selling_prices = load_problem_data(path='../data/')
    # selling_prices = change_selling_prices_format(_selling_prices)
    
    def get_ts_fleet(fleet, ts):
        # buy_df = df_init[(df_init['time_step'] == ts)]
        # move_df = df_solution[(df_solution['time_step'] == ts)]

        action_df = all_action_df[(all_action_df['time_step'] == ts)]

        # BUY NEW SERVERS
        buy_sv = []
        for i, row in action_df.iterrows(): # time_step
            for _ in range(int(row['buying_servers'])):
                sv_id = str(uuid.uuid4())
                buy_sv.append({
                    'time_step': row['time_step'],
                    'datacenter_id': row['datacenter_id'],
                    'server_generation': row['server_generation'],
                    'server_id': sv_id,
                    'action': 'buy'
                })
        solution = pd.DataFrame(buy_sv)
        # if ts > 2:
        #     prev_fleet['lifespan'] = prev_fleet['lifespan'] + 2
        # ADD REQUIRED FIELDS
        if solution.empty:
            solution = pd.DataFrame(columns=['time_step', 'datacenter_id', 'server_generation', 'server_id','action'])
        solution = solution.merge(servers, on='server_generation', how='left')
        solution = solution.merge(datacenters, on='datacenter_id', how='left')
        solution = solution.merge(selling_prices, 
                                on=['server_generation', 'latency_sensitivity'], 
                                how='left')

        # DISMISS ACTIONS
        ## DISMISS THE SERVERS THAT ARE OLDER THAN 96
        if not(fleet.empty):
            dismiss_servers = fleet[fleet['lifespan'] == 97]
            if not(dismiss_servers.empty):
                del_sv = []
                for i, row in dismiss_servers.iterrows(): # time_step
                    del_sv.append({
                        'datacenter_id': row['datacenter_id'],
                        'server_generation': row['server_generation'],
                        'server_id': row['server_id'],
                        'action': 'dismiss'
                    })
                del_df = pd.DataFrame(del_sv)
                del_df = del_df.merge(servers, on='server_generation', how='left')
                del_df = del_df.merge(datacenters, on='datacenter_id', how='left')
                del_df = del_df.merge(selling_prices, 
                                        on=['server_generation', 'latency_sensitivity'], 
                                        how='left')
                del_df['time_step'] = ts
                solution = pd.concat([solution, del_df], ignore_index=True)

        ## DISMISS SERVERS THAT ARE IN THE ACTION DF
        
        # MOVE SERVERS
        solution_after_move = solution.copy()
        remove_servers = action_df[(action_df['remove']>0)]
        moveto_servers = action_df[(action_df['move_to']>0)]

        
        if not(remove_servers.empty):
            # if not(dismiss_servers_at_ts.empty):
            #     server_id_dismiss = available_servers_dismiss['server_id'].tolist()
            # else:
            #     server_id_dismiss = []
            truck = {}
            for i, row in remove_servers.iterrows():
                src_dc = row['datacenter_id']
                src_sg = row['server_generation']
                life = row['lifespan'] - 1
                # if src_dc == 'DC1' and src_sg == 'CPU.S4':
                #     import pdb; pdb.set_trace()
                num_servers_int = int(row['remove'])
                if (row['remove'] - num_servers_int) < 0.01:
                    num_servers = int(row['remove'])
                else:
                    num_servers = math.ceil(row['remove'])
                # num_servers = int(row['remove'])
                available_servers = fleet[(fleet['lifespan'] == life) & (fleet['datacenter_id'] == src_dc) & (fleet['server_generation'] == src_sg)]
                # available_servers = available_servers[~available_servers.index.isin(server_id_dismiss)]
                available_servers = available_servers.sort_values(by='lifespan', ascending=True)
                
                # check
                if available_servers.shape[0] < num_servers:
                    print('ERROR: Not enough servers to remove')
                    print(f'Time step: {ts}')
                    print(f'{src_dc} {src_sg} {life}')
                    print(f'Available: {available_servers.shape[0]}')
                    print(f'Required: {num_servers}')



                available_servers = available_servers.head(num_servers)
                truck[(src_sg, life)] = truck.get((src_sg, life),[]) + available_servers.index.tolist()

            # Define your custom order list
            priority_servers = ['DC3', 'DC4', 'DC2', 'DC1']
            # Convert the 'datacenter_id' column to a categorical type with custom order
            moveto_servers['datacenter_id'] = pd.Categorical(moveto_servers['datacenter_id'], categories=priority_servers, ordered=True)       
            # priority_moveto_servers = moveto_servers.sort_values(by='datacenter_id', ascending=True)
            # Define your custom order list
            priority_servers = ['DC3', 'DC4', 'DC2', 'DC1']
            # Convert the 'datacenter_id' column to a categorical type with custom order
            moveto_servers['datacenter_id'] = pd.Categorical(moveto_servers['datacenter_id'], categories=priority_servers, ordered=True)       
            priority_moveto_servers = moveto_servers.sort_values(by='datacenter_id', ascending=True)
            

            for j, row2 in priority_moveto_servers.iterrows():
                dst_dc = row2['datacenter_id']
                dst_sg = row2['server_generation']
                life = row2['lifespan'] - 1

                if (row2['move_to'] - num_servers_int) < 0.01:
                    num_servers = int(row2['move_to'])
                else:
                    num_servers = math.ceil(row2['move_to'])

                available_servers = truck.get((dst_sg, life), [])
                # check
                if len(available_servers) < num_servers:
                    print('ERROR: Not enough servers to provide')
                    print(f'Time step: {ts}')
                    print(f'{src_dc} {src_sg} {life}')
                    print(f'Ready to remove: {len(available_servers)}')
                    print(f'Required: {num_servers}')



                remain_servers = []
                for sv_id in available_servers[:num_servers]:
                    item = fleet.loc[sv_id]
                    if item['datacenter_id'] == dst_dc:
                        remain_servers.append(sv_id)
                        continue
                    move_action_row = pd.DataFrame({'time_step': ts, 
                                'datacenter_id': dst_dc, 
                                'server_generation': dst_sg,
                                'server_id': item['server_id'],
                                'action': 'move',
                            }, index=[0])
                    move_action_row = move_action_row.merge(servers, on='server_generation', how='left')
                    move_action_row = move_action_row.merge(datacenters, on='datacenter_id', how='left')
                    move_action_row = move_action_row.merge(selling_prices, on=['server_generation', 'latency_sensitivity'], how='left')
                    solution_after_move = pd.concat([solution_after_move, move_action_row], ignore_index=True)
                truck[(dst_sg, life)] = remain_servers + available_servers[num_servers:]

        solution_after_dismiss = solution_after_move.copy()

        dismiss_servers_at_ts = action_df[(action_df['dismiss']>0)]

        if not(dismiss_servers_at_ts.empty):
            del_sv = []
            moved_servers = solution_after_move[(solution_after_move['action'] == 'move')]['server_id'].tolist()
            for i, row in dismiss_servers_at_ts.iterrows():
                src_dc = row['datacenter_id']
                src_sg = row['server_generation']
                life = row['lifespan'] - 1
                if row['dismiss'] < 0.01:
                    num_servers_dismiss = 0
                else:
                    num_servers_dismiss = math.ceil(row['dismiss'])
                available_servers = fleet[(fleet['lifespan'] == life) & (fleet['datacenter_id'] == src_dc) & (fleet['server_generation'] == src_sg)]
                available_servers = available_servers[~available_servers['server_id'].isin(moved_servers)]

                available_servers = available_servers.sort_values(by='lifespan', ascending=True)
                available_servers_dismiss = available_servers.head(num_servers_dismiss)
                for i, row2 in available_servers_dismiss.iterrows():
                    del_sv.append({
                            'datacenter_id': row2['datacenter_id'],
                            'server_generation': row2['server_generation'],
                            'server_id': row2['server_id'],
                            'action': 'dismiss'
                        })
                if len(del_sv) > 0:
                    del_df = pd.DataFrame(del_sv)
                    del_df = del_df.merge(servers, on='server_generation', how='left')
                    del_df = del_df.merge(datacenters, on='datacenter_id', how='left')
                    del_df = del_df.merge(selling_prices, 
                                            on=['server_generation', 'latency_sensitivity'], 
                                            how='left')
                    del_df['time_step'] = ts

                    
                    solution_after_dismiss = pd.concat([solution_after_dismiss, del_df], ignore_index=True)
                # import pdb; pdb.set_trace()

        solution_after_dismiss = solution_after_dismiss.drop_duplicates('server_id', inplace=False)
        solution_after_dismiss = solution_after_dismiss.set_index('server_id', drop=False, inplace=False)
        solution_after_dismiss = solution_after_dismiss.drop(columns='time_step', inplace=False)

        return solution_after_dismiss

    FLEET = pd.DataFrame()
    prev_FLEET = pd.DataFrame()
    for ts in tqdm(range(1, 168+1)):

        ts_fleet = get_ts_fleet(FLEET, ts)
        
        ts_solution = ts_fleet.copy()
        ts_solution['time_step'] = ts
        solution = pd.concat([solution, ts_solution], ignore_index=True)

        if ts_fleet.empty and not FLEET.empty:
            ts_fleet = FLEET
        elif ts_fleet.empty and FLEET.empty:
            continue

        # UPDATE FLEET
        prev_FLEET = FLEET.copy()
        # if ts == 110:
        #     import pdb; pdb.set_trace()
        FLEET = update_fleet(ts, FLEET, ts_fleet)
        # CHECK IF THE FLEET IS EMPTY
        if FLEET.shape[0] > 0:
            # GET THE SERVERS CAPACITY AT TIMESTEP ts
            Zf = get_capacity_by_server_generation_latency_sensitivity(FLEET)
            # CHECK CONSTRAINTS
            check_datacenter_slots_size_constraint(FLEET)

            FLEET = put_fleet_on_hold(FLEET)
        

    solution = solution[['time_step', 'action', 'server_id', 'datacenter_id', 'server_generation']]
    final_solution = solution.to_dict(orient='records')
    final_solution = {
        'fleet': final_solution,
        'pricing_strategy': [
            {
            "time_step": 1,
            "latency_sensitivity": "low",
            "server_generation": "CPU.S1",
            "price": 10
            }
        ]
    }
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(final_solution, out, ensure_ascii=False, indent=4)
    print(f'Submission saved to {path}')

def main():
    all_action_df = pd.read_csv('all_action_df.csv')
    df_to_submissions(all_action_df, path=f'{seed}.json')
    done_flag.set()  # Signal that my_code is done

def cooking_rabbit_animation():
    global done_flag
    while not done_flag.is_set():  # Keep running until my_code is done
        for frame in rabbit_frames:
            clear()
            print(frame)
            time.sleep(1.0)  # Pause between frames
            if done_flag.is_set():  # Check flag after each frame
                break
    
    # After my_code is done, show the finished frame
    clear()
    print(finished_frame)


if __name__ == "__main__":
    # Create threads for the animation and your code
    # rabbit_thread = threading.Thread(target=cooking_rabbit_animation)
    # my_code_thread = threading.Thread(target=main)

    # # Start both threads
    # rabbit_thread.start()
    # my_code_thread.start()

    # # Wait for both threads to finish
    # rabbit_thread.join()
    # my_code_thread.join()
    main()