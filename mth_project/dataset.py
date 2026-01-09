import os
import csv
import pandas as pd
class Dataset:
    def __init__(self, directory):
        self.directory = directory
    def get_device(self, device_name):
        """
        Returns pd.DataFrame for particular network device in the format
        device = pd.DataFrame(
        {
        'timestamp': pd.to_datetime([]),
        'direction': ['sent', 'received']
        'port': ['port1', 'port2', 'port3', 'port6', 'port7']
        'value': []
        }
        )
        """
        #device = pd.DataFrame(columns=['timestamp', 'direction', 'port', 'value'])
        device = pd.DataFrame()
        for filename in os.listdir(self.directory):
            if filename.startswith(device_name) and filename.endswith(".csv"):
                filepath = os.path.join(self.directory, filename)
                try:
                    df = pd.read_csv(filepath, delimiter='\t', header=0, names=['name', 'timestamp', 'value'], skiprows=[1])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df_network_traffic = df[df['name'].str.lower().str.contains('bits received|bits sent')].copy()
                    df_network_traffic.loc[:,'direction'] = df_network_traffic['name'].apply(lambda x: 'sent' if 'bits sent' in x.lower() else 'received')
                    
                    ports = {}
                    for port in df_network_traffic['name'].unique():
                        port_lower = port.lower()
                        if 'gi1/' not in port_lower:
                            continue
                        try:
                            port_number = port_lower.split('gi1/')[1].split('(')[0].strip()
                            port_name = port_lower.split('_gi1/')[1].split('(')[1].split(')')[0].strip() if '_gi1/' in port_lower else 'unknown'
                            
                            if port_number not in ports:
                                ports[port_number] = port_name
                        except (IndexError, AttributeError) as e:
                            print(f"Warning: Could not parse port from {port}: {e}")
                            continue
                    
                    df_network_traffic.loc[:,'port'] = df_network_traffic['name'].apply(lambda x: x.lower().split('gi1/')[1].split('(')[0].strip() if 'gi1/' in x.lower() else 'unknown')
                
                    df_network_traffic = df_network_traffic[['timestamp', 'direction', 'port', 'value']]
                    device = pd.concat([device, df_network_traffic], ignore_index=True)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        device = device.sort_values(by='timestamp').reset_index(drop=True)
        return device, ports

    def get_device_names(self):
        """
        Processes CSV files in the given directory and returns list of names of devices.}
        """
        device_names = []
        
        for filename in os.listdir(self.directory):
            if filename.endswith(".csv"):
                hostname = filename.split("_")[0]
                device_names.append(hostname)
        return device_names
