import os
import utils
import youtokentome as yttm

def train_tokenizer():
    args, unk = utils.get_args()

    train_dataset, _, _ = utils.load_llm_dataset(args.train_dataset, splits=('train'))

    # Extract the text data from the train dataset
    train_texts = train_dataset["content"]

    # Save the train texts to a file
    with open("train_data.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(train_texts))

    model_path = os.path.join('llm', 'runs', args.run_name, 'bpe.model')

    yttm.BPE.train(data="train_data.txt", vocab_size=args.vocab_size, model=model_path)

    bpe_model = yttm.BPE(model=model_path)

    print(bpe_model.vocab())
    print(bpe_model.encode("Hello, how are you?"))

if __name__ == '__main__':
    train_tokenizer()
