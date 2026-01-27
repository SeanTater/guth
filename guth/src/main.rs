use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "guth")]
#[command(about = "Rust rewrite of pocket-tts", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Say {
        text: String,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long)]
        output: Option<String>,
    },
    Voices,
    Models,
    Download {
        model: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Say { text, voice, output } => {
            println!("Say: {text}");
            if let Some(voice) = voice {
                println!("Voice: {voice}");
            }
            if let Some(output) = output {
                println!("Output: {output}");
            }
        }
        Commands::Voices => {
            println!("List voices (todo)");
        }
        Commands::Models => {
            println!("List models (todo)");
        }
        Commands::Download { model } => {
            println!("Download model: {model}");
        }
    }

    Ok(())
}
