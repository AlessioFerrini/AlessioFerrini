import logging
import src.experiments

# set up logging
logging.root.setLevel(logging.INFO)

def main():
    src.experiments.vascular_sprouting()

if __name__ == "__main__":
    main()