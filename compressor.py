import os
import zipfile
from zipfile import ZipFile
import re
from transformers import AlbertTokenizer, GPT2LMHeadModel
import torch

PREDICT_WINDOW = 10000
NUM_WORDS = 100
SHOW_TOKENS = True
SHOW_PREDICTION = True
COMPRESS_LEVEL = 9
COMPRESS_MODE = zipfile.ZIP_DEFLATED

class TextCompressor:
    def __init__(self, model_name):
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def compress(self, input_text):
        compressed_tokens = []
        current_text = ""

        # Use regular expression to split the input text into tokens with spaces as separators
        pattern = r'\d+|[а-яА-ЯїЇєЄіIґҐ]+\'?+[а-яА-ЯїЇєЄіIґҐ]*|[^а-яА-Я0-с9їЇєЄіIґҐ]'
        no_letter_or_didgit_pattern = r'[^а-яА-Я0-9їЇєЄіIґҐ]'
        tokens = re.findall(pattern, input_text)
        if SHOW_TOKENS:
            print(tokens)

        if tokens:
            # Handle the first word separately
            first_word = tokens[0]
            compressed_tokens.append(first_word)
            current_text += first_word
            tokens = tokens[1:]

        for token in tokens:
            if re.fullmatch(no_letter_or_didgit_pattern, token):
                # If the token is spaces or a special symbol, append it as is
                compressed_tokens.append(token)
            elif token.isdigit():
                # If the token is a number, simply append it with a prefix "nu"
                compressed_tokens.append("<n" + token+">")
            else:
                # Predict the next words based on the current text
                prediction = self.predict_next(current_text, NUM_WORDS)
                if SHOW_PREDICTION:
                        print(prediction)

                if token in prediction:
                    # If the token is predicted, append its index as a number
                    index = prediction.index(token)
                    compressed_tokens.append(str(index))
                    if SHOW_PREDICTION:
                        print(index)
                else:
                    # If the token is not predicted, append it as is
                    compressed_tokens.append(token)
                    if SHOW_PREDICTION:
                        print(token)

            # Update the current_text
            current_text += token

        # Join the compressed tokens into a single string
        compressed_text = "".join(compressed_tokens)
        return compressed_text

    def decompress(self, compressed_text):
        decompressed_tokens = []
        current_text = ""
        add_number = False  # Flag to indicate the next number should be added as it is

        # Use regular expression to split the compressed text into tokens
        pattern = r'<n\d+>|\d+|[а-яА-ЯїЇєЄіIґҐ]+\'?+[а-яА-ЯїЇєЄіIґҐ]*|[^а-яА-Я0-9їЇєЄіIґҐ]'
        number_pattern = r'<n\d+>'
        tokens = re.findall(pattern, compressed_text)
        if SHOW_TOKENS:
            print(tokens)

        for token in tokens:
            if re.fullmatch(number_pattern, token):
                # If the token matches the number pattern add it as a regular number
                token = token[2:len(token)-1]
                decompressed_tokens.append(token)
            elif token.isdigit():
                # Convert the number to an integer
                word_index = int(token)
                # Predict the next words based on the current text
                prediction = self.predict_next(current_text, NUM_WORDS)

                # Use the predicted word at the specified index
                if word_index < len(prediction):
                    word = prediction[word_index]
                    # Check if the word is a special character and adjust spacing accordingly
                    decompressed_tokens.append(word)
                else:
                    # If the index is out of range, append the number itself
                    decompressed_tokens.append(token)
            else:
                # If the token is not a number, simply append it to the current text
                decompressed_tokens.append(token)

            # Update the current_text
            current_text = "".join(decompressed_tokens)

        # Join the decompressed tokens into a single string
        decompressed_text = "".join(decompressed_tokens)
        return decompressed_text

    def predict_next(self, input_text, num_words):
        input_text = input_text[-PREDICT_WINDOW:]
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt')

        # Generate probabilities for the next word
        with torch.no_grad():
            logits = self.model(input_ids).logits

        # Get the last token's probabilities
        next_word_logits = logits[:, -1, :]

        # Get the top num_words words with the highest probabilities
        top_next_word_indices = torch.topk(next_word_logits, k=num_words, dim=-1).indices[0]
        top_next_words = [self.tokenizer.decode(idx.item()) for idx in top_next_word_indices]
        if SHOW_PREDICTION:
            print(top_next_words)
        return top_next_words


# Example usage:
model_name = "Tereveni-AI/gpt2-124M-uk-fiction"
text_compressor = TextCompressor(model_name)
print(type(COMPRESS_MODE))

while (True):
    command = input('>>>')
    if command == "stop":
        break
    if command == "help":
        print("stop")
        print("zip file_name")
        print("compress file_name")
        print("zip_compress file_name")
        print("decompress file_name")
        continue
    filename = input('Enter a file name: ')
    try:
        in_file = open(filename, 'r')
    except FileNotFoundError:
        print("FileNotFoundError: No such file or directory", filename)
        continue
    except PermissionError:
        print("Permission Denied Error: Access is denied")
        continue
    else:
        input_text = in_file.read()
        print("Input text: ", input_text)
    finally:
        in_file.close()

    input_size = len(input_text)
    if command == "zip":
        bold_filename, file_extension = os.path.splitext(filename)
        with ZipFile(bold_filename + '.zip', 'w', compression = COMPRESS_MODE, compresslevel=COMPRESS_LEVEL) as zip:
            zip.writestr(filename, input_text)
            print("File was zipped")

    if command == "compress":
        with open("(compressed)" + filename, 'w') as out_file:
            compressed_text = text_compressor.compress(input_text)
            print("Compressed text: ", compressed_text)
            out_file.write(compressed_text)
            output_size = len(compressed_text)
            print("Compressing status: {:%}".format((input_size - output_size)/input_size))
    if command == "zip_compress":
        bold_filename, file_extension = os.path.splitext(filename)
        with ZipFile(bold_filename + '.zip', 'w', compression = COMPRESS_MODE, compresslevel=COMPRESS_LEVEL) as zip:
            compressed_text = text_compressor.compress(input_text)
            print("Compressed text: ", compressed_text)
            output_size = len(compressed_text)
            zip.writestr("(compressed)" + filename, compressed_text)
            print("Compressing status: {:%}".format((input_size - output_size) / input_size))

    if command == "decompress":
        with open("(decompressed)" + filename, 'w') as out_file:
            decompressed_text = text_compressor.decompress(input_text)
            print("Decompressed text: ", decompressed_text)
            out_file.write(decompressed_text)

