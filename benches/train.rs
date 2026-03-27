use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use rustbpe::Tokenizer;

fn repeated_corpus(lines: usize) -> Vec<String> {
    let templates = [
        "the quick brown fox jumps over the lazy dog 0123456789",
        "byte pair encoding likes repeated chunks and repeated punctuation!!!",
        "rustbpe trains once and then exports mergeable ranks to tiktoken",
        "naive heap churn should not dominate the incremental merge scheduler",
        "bonjour le monde tokenizer training should still handle unicode accents",
        "symbols []{}() <> +=- keep regex splitting honest across categories",
    ];

    (0..lines)
        .map(|i| {
            format!(
                "{} | sample:{} | bucket:{}",
                templates[i % templates.len()],
                i % 97,
                i % 11
            )
        })
        .collect()
}

fn bench_training(c: &mut Criterion) {
    let medium = repeated_corpus(4_096);
    let large = repeated_corpus(12_288);

    let mut group = c.benchmark_group("train_from_texts");
    group.sample_size(10);

    group.throughput(Throughput::Elements(medium.len() as u64));
    group.bench_function("medium_vocab_1024", |b| {
        b.iter_batched(
            Tokenizer::new,
            |mut tok| {
                tok.train_from_texts(black_box(&medium), 1_024, None)
                    .expect("benchmark training should succeed");
                black_box(tok.vocab_size());
            },
            BatchSize::SmallInput,
        );
    });

    group.throughput(Throughput::Elements(large.len() as u64));
    group.bench_function("large_vocab_2048", |b| {
        b.iter_batched(
            Tokenizer::new,
            |mut tok| {
                tok.train_from_texts(black_box(&large), 2_048, None)
                    .expect("benchmark training should succeed");
                black_box(tok.vocab_size());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_training);
criterion_main!(benches);
