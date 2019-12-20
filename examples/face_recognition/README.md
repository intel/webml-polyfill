# WebNN API Face Recognition Example
This example loads Face Detection models (trained by WIDERFace) and Face Recognition models (include facenet model and face-reidentification model), constructs and inferences them by WebNN API.

For face-reidentification model, The outputs on different images are comparable in cosine distance and cosine distance threshold is about 0.8 (based [google book - page 235](https://books.google.co.kr/books?id=UB1tDwAAQBAJ&pg=PA235&lpg=PA235&dq=LFW+cosine+threshold&source=bl&ots=DVTv9oVGbN&sig=ACfU3U0f3Imeto2SjfW8JwSu5RjWiMpI2A&hl=en&sa=X&ved=2ahUKEwijhcDGzpjmAhVRL6YKHQYFAvAQ6AEwA3oECAkQAQ#v=onepage&q=LFW%20cosine%20threshold&f=false)).

For facenet model, The outputs on different images are comparable in euclidean distance and euclidean distance threshold is about 1.26 (based [paper - 5.6](https://arxiv.org/pdf/1503.03832.pdf)).

## Download Model
Before launching this example, you need to download the model. Please check out [README.md](model/README.md) in model folder for details.
