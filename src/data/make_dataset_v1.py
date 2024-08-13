import pandas as pd

def load_data(data_path):
    # Load your dataset from a given path
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(data_path)
    return df

def clean_data(df):
    df.insert(loc=3, column='sector', value=df['property_name'].str.split('in').str.get(1).str.replace('Gurgaon', '').str.strip())
    replacements = {
        'dharam colony': 'sector 12', 'krishna colony': 'sector 7', 'suncity': 'sector 54',
        'prem nagar': 'sector 13', 'mg road': 'sector 28', 'gandhi nagar': 'sector 28',
        'laxmi garden': 'sector 11', 'shakti nagar': 'sector 11', 'baldev nagar': 'sector 7',
        'shivpuri': 'sector 7', 'garhi harsaru': 'sector 17', 'imt manesar': 'manesar',
        'adarsh nagar': 'sector 12', 'shivaji nagar': 'sector 11', 'bhim nagar': 'sector 6',
        'madanpuri': 'sector 7', 'saraswati vihar': 'sector 28', 'arjun nagar': 'sector 8',
        'ravi nagar': 'sector 9', 'vishnu garden': 'sector 105', 'bhondsi': 'sector 11',
        'surya vihar': 'sector 21', 'devilal colony': 'sector 9', 'valley view estate': 'gwal pahari',
        'mehrauli road': 'sector 14', 'jyoti park': 'sector 7', 'ansal plaza': 'sector 23',
        'dayanand colony': 'sector 6', 'sushant lok phase 2': 'sector 55', 'chakkarpur': 'sector 28',
        'greenwood city': 'sector 45', 'subhash nagar': 'sector 12', 'sohna road road': 'sohna road',
        'malibu town': 'sector 47', 'surat nagar 1': 'sector 104', 'new colony': 'sector 7',
        'mianwali colony': 'sector 12', 'jacobpura': 'sector 12', 'rajiv nagar': 'sector 13',
        'ashok vihar': 'sector 3', 'dlf phase 1': 'sector 26', 'nirvana country': 'sector 50',
        'palam vihar': 'sector 2', 'dlf phase 2': 'sector 25', 'sushant lok phase 1': 'sector 43',
        'laxman vihar': 'sector 4', 'dlf phase 4': 'sector 28', 'dlf phase 3': 'sector 24',
        'sushant lok phase 3': 'sector 57', 'dlf phase 5': 'sector 43', 'rajendra park': 'sector 105',
        'uppals southend': 'sector 49', 'sohna': 'sohna road', 'ashok vihar phase 3 extension': 'sector 5',
        'south city 1': 'sector 41', 'ashok vihar phase 2': 'sector 5', 'sector 95a': 'sector 95',
        'sector 23a': 'sector 23', 'sector 12a': 'sector 12', 'sector 3a': 'sector 3',
        'sector 110 a': 'sector 110', 'patel nagar': 'sector 15', 'a block sector 43': 'sector 43',
        'maruti kunj': 'sector 12', 'b block sector 43': 'sector 43', 'sector-33 sohna road': 'sector 33',
        'sector 1 manesar': 'manesar', 'sector 4 phase 2': 'sector 4', 'sector 1a manesar': 'manesar',
        'c block sector 43': 'sector 43', 'sector 89 a': 'sector 89', 'sector 2 extension': 'sector 2',
        'sector 36 sohna road': 'sector 36'
    }

    df['sector'] = df['sector'].replace(replacements)
    
    a = df['sector'].value_counts()[df['sector'].value_counts() >= 3]
    df = df[df['sector'].isin(a.index)]
    
    df['sector'] = df['sector'].replace({
        'sector 95a': 'sector 95', 'sector 23a': 'sector 23', 'sector 12a': 'sector 12',
        'sector 3a': 'sector 3', 'sector 110 a': 'sector 110', 'patel nagar': 'sector 15',
        'a block sector 43': 'sector 43', 'maruti kunj': 'sector 12', 'b block sector 43': 'sector 43',
        'sector-33 sohna road': 'sector 33', 'sector 1 manesar': 'manesar', 'sector 4 phase 2': 'sector 4',
        'sector 1a manesar': 'manesar', 'c block sector 43': 'sector 43', 'sector 89 a': 'sector 89',
        'sector 2 extension': 'sector 2', 'sector 36 sohna road': 'sector 36'
    })
    
    df.loc[955, 'sector'] = 'sector 37'
    df.loc[2800, 'sector'] = 'sector 92'
    df.loc[2838, 'sector'] = 'sector 90'
    df.loc[2857, 'sector'] = 'sector 76'
    df.loc[[311, 1072, 1486, 3040, 3875], 'sector'] = 'sector 110'
    
    df.drop(columns=['property_name', 'address', 'description', 'rating'], inplace=True)
    return df

def make_dataset_v1_to_file(df):
    output_path = 'E:\\Work files\\Real_state\\data\\processed\\gurgaon_properties_cleaned_v1.csv'
    df.to_csv(output_path, index=False)

# Load data
df = load_data("E:\\Work files\\Real_state\\data\\processed\\gurgaon_properties.csv")

# Clean data
df = clean_data(df)

# Save cleaned data to file
make_dataset_v1_to_file(df)
