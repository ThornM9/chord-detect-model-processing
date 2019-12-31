#!/usr/bin/env node

/**
 * This script will process an entire directory of wav files and output a data.csv file that contains
 * the chroma vector and chord label for a wav file.
 */
const fs = require("fs");
const WavDecoder = require("wav-decoder");
const Meyda = require('meyda');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

const OUTPUT_CSV_NAME = "validation_data.csv";

const csvWriter = createCsvWriter({
  path: OUTPUT_CSV_NAME,
  header: [
    {id: 'c', title: 'c'},
    {id: 'c#', title: 'c#'},
    {id: 'd', title: 'd'},
    {id: 'd#', title: 'd#'},
    {id: 'e', title: 'e'},
    {id: 'f', title: 'f'},
    {id: 'f#', title: 'f#'},
    {id: 'g', title: 'g'},
    {id: 'g#', title: 'g#'},
    {id: 'a', title: 'a'},
    {id: 'a#', title: 'a#'},
    {id: 'b', title: 'b'},
    {id: 'chord', title: 'chord'},
  ]
});

let chord_names = ['a', 'am', 'bm', 'c', 'd', 'dm', 'e', 'em', 'f', 'g'];

// process all wav files in this directory
const BASE_INPUT_DIRECTORY = "C:\\Users\\Thornton\\Desktop\\Home\\Coding\\guitar_chords_ai\\processing\\dataset\\Other_Instruments\\Guitar\\";

// function to read a wav file and return the signal
const readFile = (filepath) => {
    return new Promise((resolve, reject) => {
        fs.readFile(filepath, (err, buffer) => {
        if (err) {
            return reject(err);
        }
        return resolve(buffer);
        });
    });
};

function calculateAverageChroma(chromas) {
    let avgChroma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < chromas.length; i++) {
        for (let j = 0; j < chromas[i].length; j++) {
            avgChroma[j] += chromas[i][j];
        }
    }

    for (let i = 0; i < avgChroma.length; i++) {
        avgChroma[i] = avgChroma[i] / chromas.length;
    }
    return avgChroma;
}

// configurations
const BUFFER_SIZE = 16384
Meyda.bufferSize = BUFFER_SIZE;
Meyda.sampleRate = 44100;

let promises = [];
let files_remaining = 0;
function processFile(filepath, chord) {
    // read and decode the file
    let promise = readFile(filepath).then((buffer) => {
        return WavDecoder.decode(buffer);
    }).then(function(audioData) {
        Meyda.sampleRate = audioData.sampleRate;
        let samplesRemaining = audioData.channelData[0].length;
        let i = 0;
        let chromas = []
        while (samplesRemaining > BUFFER_SIZE) {
            // extract buffer from signal
            let signal = audioData.channelData[0].slice(i * BUFFER_SIZE, i * BUFFER_SIZE + BUFFER_SIZE);

            // analyse signal with meyda
            let chroma = Meyda.extract("chroma", signal);
            chromas.push(chroma);
            i += 1;
            samplesRemaining -= BUFFER_SIZE;
        }
        // TODO maybe normalise the chroma too
        let avgChroma = calculateAverageChroma(chromas)
        // add to data to export
        csv_data.push({
            "c": avgChroma[0],
            "c#": avgChroma[1],
            "d": avgChroma[2],
            "d#": avgChroma[3],
            "e": avgChroma[4],
            "f": avgChroma[5],
            "f#": avgChroma[6],
            "g": avgChroma[7],
            "g#": avgChroma[8],
            "a": avgChroma[9],
            "a#": avgChroma[10],
            "b": avgChroma[11],
            "chord": chord_names.indexOf(chord),
        })
        files_remaining -= 1;
    });
    // track the promises
    promises.push(promise);
}
let csv_data = [];
// extract chroma vectors
for (let chord of chord_names) {
    let input_directory = BASE_INPUT_DIRECTORY + chord + "\\"
    let filenames = fs.readdirSync(input_directory);
    files_remaining += filenames.length;
    for (let file of filenames) {
        processFile(input_directory + file, chord);
    }

}
console.log(`Total number of files to process: ${files_remaining}`);

setInterval(() => {
    console.log(`Files Remaining: ${files_remaining}`);
}, 2000)

Promise.all(promises).then(function(values) {
    csvWriter
    .writeRecords(csv_data)
    .then(()=> {
        console.log('Finished!');
        process.exit();
    });
});