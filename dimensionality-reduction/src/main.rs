/*
A comprehensive toolkit for Statistical Learning in Rust.
Linfa is a comprehensive toolkit for statistical learning, providing algorithms for optimal model and density estimation.
*/
use linfa::prelude::*;
use linfa_reduction::Pca;

fn main() {
    let dataset = linfa_datasets::diabetes(1599, 11, 1).split_with_ratio(0.8);
    let embedding: Pca<f64> = Pca::params(2).fit(&dataset).unwrap();
    // let embedding = embedding.predict(&dataset);

    // println!("Reduced Dataset: {:?}", dataset);

    dbg!(dataset);
}
