<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;

const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Iris Flower Classifier using K Nearest Neighbors              ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

#Import data
$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

#Feature data
$samples = $reader->getRecords([
    'septal-length', 'sepal-with', 'petal-length', 'petal-width',
]);

#Target Labels
$labels = $reader->fetchColumn('class');

#Preparing the dataset (data_point, target_label)
$dataset = Labeled::fromIterator($samples, $labels);
$dataset->apply(new NumericStringConverter());

#Splitting dataset into train and test data
[$training, $testing] = $dataset->randomize()->stratifiedSplit(0.80);

#Initialsing KNN model
$estimator = new KNearestNeighbors(5, new Euclidean());

#Training the model
$estimator->train($training);

#Predicting on test data
$predictions = $estimator->predict($testing);

#Comparing predictions with original class/labels
$report = new MulticlassBreakdown();
$results = $report->generate($predictions, $testing->labels());

#Write results to a file
file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to ' . REPORT_FILE . PHP_EOL;
