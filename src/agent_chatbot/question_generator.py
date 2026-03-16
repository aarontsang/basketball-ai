"""
This file contains functions to generate a local dataset of question-answer pairs based on the 
NBA Wikipedia corpus and to generate specific questions about player statistics. The dataset can 
be used for training or evaluating the chatbot agent. The functions utilize a local Ollama model 
to create structured JSON outputs that can be easily parsed and stored.
"""
import json
import random
from llama_index.llms.ollama import Ollama
from stats_query_handler import get_player_stats, get_team_info


# 1. Initialize your local Ollama model
# Ensure Ollama is running in the background (ollama serve)
llm = Ollama(model="llama3", request_timeout=120.0, format="json")

def generate_local_dataset(json_path, output_path="golden_dataset.json", num_pairs=20):
    # Load your existing corpus
    try:
        with open(json_path, "r") as f:
            corpus = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return

    titles = list(corpus.keys())
    golden_dataset = []

    print(f"--- Starting Local Generation for {num_pairs} Q&A pairs ---")

    for i in range(num_pairs):
        topic = random.choice(titles)
        context = corpus[topic]

        # Construct a prompt that forces the LLM to output clean JSON
        prompt = f"""
        Context: {context}
        ---
        Generate a JSON object with 'question' and 'answer' based on the NBA context above.
        The JSON must look exactly like this: {{"question": "...", "answer": "..."}}
        Begin your response with '{{' and end with '}}'.
        """

        try:
            print(f"Generating pair {i+1}/{num_pairs} from topic: {topic}...")
            response = llm.complete(prompt)
            raw_text = response.text.strip()
            
            # Locate the JSON block
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            
            if start == -1 or end == 0:
                print(f"Skipping {i+1}: No JSON braces found in response.")
                continue
                
            json_str = raw_text[start:end]
            qa_pair = json.loads(json_str)
            
            # Success!
            qa_pair["source"] = topic
            golden_dataset.append(qa_pair)
            
        except json.JSONDecodeError:
            print(f"Skipping {i+1}: Model produced malformed JSON: {raw_text[:50]}...")
        except Exception as e:
            print(f"Skipping {i+1} due to error: {e}")

    # Save the results
    with open(output_path, "w") as f:
        json.dump(golden_dataset, f, indent=4)
    
    print(f"\nDone! Dataset saved to {output_path}")


def generate_stats_question(player_name, season="2023-24"):
    context = get_player_stats(player_name, scope="season", competition="regular", view="per_game", season=season)
    prompt = f"""
    Context: {context}
    Generate a JSON object with 'question' and 'answer' about the NBA player {player_name} for the {season} season.
    The question should ask for specific statistics (e.g., points per game, rebounds, etc.) and the answer should provide those stats in a concise format.
    The JSON must look exactly like this: {{"question": "...", "answer": "..."}}
    Begin your response with '{{' and end with '}}'.
    """
    response = llm.complete(prompt)
    return response.text.strip()

def stats_questions_generation_script(player_names, season=None, output_path="stats_questions.json"):
    stats_dataset = []
    
    for player in player_names:
        print(f"Generating stats question for {player}...")
        qa_pair = generate_stats_question(player, season)
        
        try:
            start = qa_pair.find("{")
            end = qa_pair.rfind("}") + 1
            
            if start == -1 or end == 0:
                print(f"Skipping {player}: No JSON braces found in response.")
                continue
            
            json_str = qa_pair[start:end]
            stats_dataset.append(json.loads(json_str))
        except json.JSONDecodeError:
            print(f"Skipping {player}: Model produced malformed JSON: {qa_pair[:50]}...")
        except Exception as e:
            print(f"Skipping {player} due to error: {e}")

    with open(output_path, "w") as f:
        json.dump(stats_dataset, f, indent=4)
    
    print(f"\nDone! Stats questions saved to {output_path}")

if __name__ == "__main__":
    # Ensure this filename matches your actual JSON file
    # generate_local_dataset("nba_wikipedia_corpus.json", num_pairs=20)
    # Example player list for stats question generation
    players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo", 
               "Kevin Durant", "Nikola Jokic", "Joel Embiid", "Luka Doncic",
               "Jayson Tatum", "Ja Morant", "Devin Booker",
               "Anthony Davis", "Kawhi Leonard", "Damian Lillard", "Zion Williamson",
               "Jimmy Butler", "Bradley Beal", "Trae Young", "Karl-Anthony Towns", "Chris Paul", 
               "Russell Westbrook", "Paul George"]
    stats_questions_generation_script(players, season=None)
    