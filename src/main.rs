use iced::{
    time, alignment, executor, Application, Element,
    Length, Settings, Subscription, Theme, Command
};
use iced::widget::{
    button, container, Column, Row, row, text
};
use rand::Rng;
use std::collections::HashMap;
use std::default::Default;
use std::fmt;
use std::time::{Duration, Instant};
use rand::prelude::IndexedRandom;

const TRAIN_EPISODE: usize = 100000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    Empty,
    X,
    O,
}

#[derive(Debug, Clone)]
struct Player {
    marker: Cell, // X or O
}

// impl Player {
//     fn new(marker: Cell) -> Self {
//         Player { marker }
//     }
//     fn opponent(&self) -> Self {
//         if self.marker == Cell::X {
//             Player::new(Cell::O)
//         } else if self.marker == Cell::O {
//             Player::new(Cell::X)
//         } else { Player::new(Cell::Empty) }
//     }
// }

#[derive(Debug, Clone)]
struct MoveStatus {
    move_successful: bool,
    game_over: bool,
}

#[derive(Debug, Clone)]
struct Board {
    grid: [[Cell; 3]; 3],
    players: [Player; 2],
    current_player: usize,
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Cell::Empty => write!(f, "-"),
            Cell::X => write!(f, "X"),
            Cell::O => write!(f, "O"),
        }
    }
}

impl MoveStatus {
    fn new(move_successful: bool, game_over: bool) -> Self {
        MoveStatus {
            move_successful,
            game_over
        }
    }
}

impl Board {
    fn new() -> Self {
        Board {
            grid: [[Cell::Empty; 3]; 3],
            players: [Player { marker: Cell::X }, Player { marker: Cell::O }],
            current_player: rand::rng().random_range(0..=1),
        }
    }
    fn get_current_player(&self) -> &Player {
        &self.players[self.current_player]
    }
    fn make_move(&mut self, row: usize, col: usize) -> (MoveStatus, Option<Cell>) {
        if self.grid[row][col] == Cell::Empty {
            let current_player = &self.players[self.current_player];
            self.grid[row][col] = current_player.marker;
            self.switch_turn();
            let (game_over, winner) = self.is_game_over();
            (MoveStatus::new(true, game_over), winner)
        } else {
            (MoveStatus::new(false, false), None)
        }
    }
    fn switch_turn(&mut self) {
        self.current_player = 1 - self.current_player;
    }
    fn check_winner(&self) -> Option<Cell> {
        for i in 0..3 {
            if self.grid[i][0] != Cell::Empty
                && self.grid[i][1] == self.grid[i][0]
                && self.grid[i][2] == self.grid[i][1]
            {
                return Some(self.grid[i][0]);
            }
            if self.grid[0][i] != Cell::Empty
                && self.grid[1][i] == self.grid[0][i]
                && self.grid[2][i] == self.grid[1][i]
            {
                return Some(self.grid[0][i]);
            }
        }
        if self.grid[0][0] != Cell::Empty
            && self.grid[1][1] == self.grid[0][0]
            && self.grid[2][2] == self.grid[1][1]
        {
            return Some(self.grid[0][0]);
        }
        if self.grid[0][2] != Cell::Empty
            && self.grid[1][1] == self.grid[0][2]
            && self.grid[2][0] == self.grid[1][1]
        {
            return Some(self.grid[0][2]);
        }
        None
    }

    fn is_draw(&self) -> bool {
        self.grid.iter().flatten().all(|&x| x != Cell::Empty)
    }

    fn is_game_over(&self) -> (bool, Option<Cell>) {
        let winner = self.check_winner();
        (winner.is_some() || self.is_draw(), winner)
    }

    fn board_state(&self) -> String {
        self.grid
            .iter()
            .flatten()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>()
            .join("")
    }

    fn available_moves(&self) -> Vec<(usize, usize)> {
        let possible_moves = self
            .grid
            .iter()
            .flatten()
            .map(|&x| match x {
                Cell::Empty => 1,
                _ => 0,
            })
            .collect::<Vec<usize>>();

        let total_moves = vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ];
        total_moves
            .into_iter()
            .zip(possible_moves.into_iter())
            .filter(|&(_, flag)| flag==1)
            .map(|(x, _)| x)
            .collect()
    }
    fn find_blocking_move(&self) -> Option<(usize, usize)> {
        for pos in self.available_moves() {
            let mut temp_board = self.clone();
            temp_board.switch_turn();
            let current_player_marker = temp_board.get_current_player().marker;
            let (_, winner) = temp_board.make_move(pos.0,pos.1);

            if winner.is_some(){
                if winner.unwrap() == current_player_marker {
                    return Some(pos);
                }
            }
        }
        None
    }
    fn reset(&mut self) {
        let mut rng = rand::rng();
        self.grid = [[Cell::Empty; 3]; 3];
        self.current_player = rng.random_range(0..=1);
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.board_state())
    }
}

#[derive(Debug)]
struct QLearningAgent {
    q_table: HashMap<String, HashMap<String, f64>>,
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    train: bool
}

impl QLearningAgent {
    fn new(alpha: f64, gamma: f64, epsilon: f64) -> Self {
        QLearningAgent {
            q_table: HashMap::new(),
            alpha,
            gamma,
            epsilon,
            train: true
        }
    }
    fn get_q_value(&mut self, state: &str, action: &str) -> f64 {
        *self
            .q_table
            .entry(state.to_string())
            .or_insert(HashMap::new())
            .entry(action.to_string())
            .or_insert(0.0)
    }

    fn update_q_value(&mut self, state: &str, action: &str, reward: f64, next_state: &str) {
        let max_q_next = self
            .q_table
            .get(next_state)
            .map(|actions| actions.values().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0);
        let old_q_value = self.get_q_value(state, action);
        let new_q_value =
            old_q_value + self.alpha * (reward + self.gamma * max_q_next - old_q_value);
        self.q_table
            .entry(state.to_string())
            .or_insert(HashMap::new())
            .insert(action.to_string(), new_q_value);
    }

    fn choose_action(&mut self, state: &str, available_moves: &Vec<(usize, usize)>, blocking_move: Option<(usize, usize)>) -> ((usize, usize),bool) {
        if blocking_move.is_some() && self.train {
            (blocking_move.unwrap(), true)
        } else {
        let mut rng = rand::rng();
        if (rng.random::<f64>() < self.epsilon) && self.train {
            (*available_moves.choose(&mut rng).unwrap(),false)
        } else {
            let q_values = self.q_table.get_mut(state);
            if let Some(actions) = q_values {
                let best_action = available_moves
                    .iter()
                    .max_by(|&a, &b| {
                        let str_a = format!("{},{}", a.0,a.1);
                        let str_b = format!("{},{}", b.0,b.1);
                        let q_a = actions.get(&str_a).unwrap_or(&0.0);
                        let q_b = actions.get(&str_b).unwrap_or(&0.0);
                        q_a.partial_cmp(&q_b).unwrap()
                    }).unwrap();
                (*best_action, false)
            } else {
                (*available_moves.choose(&mut rng).unwrap(),false)
            }
        }
        }
    }

}

fn train_q_learning(agent: &mut QLearningAgent, episode: usize) {
    for _ in 0..episode {
        let mut game = Board::new();
        loop {
            let (game_over,_) = game.is_game_over();
            if game_over {
                break;
            };
            let state = game.board_state();
            let moves = game.available_moves();
            let blocking_move = game.find_blocking_move();
            let current_player = game.get_current_player().clone();
            let (action,is_blocking_move) = agent.choose_action(&state, &moves, blocking_move);
            let action_hash = format!("{},{}", action.0, action.1);
            game.make_move(action.0,action.1);
            let reward = if game.check_winner().unwrap_or(Cell::Empty) == current_player.marker { 1.0 }
                                else if is_blocking_move {1.0}
                                else if game.is_draw(){0.5}
                                else {0.0};
            let next_state = game.board_state();
            agent.update_q_value(&state,&action_hash,reward,&next_state);
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone,PartialEq)]
enum GameMode {
    PvP,
    PvA
}

#[derive(Debug, Clone)]
enum Message {
    CellClicked(usize, usize),
    ResetGame,
    AIMove,
    Tick,
    SetGameMode(GameMode)
}

struct TicTacToeApp {
    board: Board,
    game_over: bool,
    winner: Option<Cell>,
    ai_agent: QLearningAgent,
    game_mode: GameMode,
    ai_thinking: bool,
    ai_turn_start: Option<Instant>,
}

impl Application for TicTacToeApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let mut agent: QLearningAgent = QLearningAgent::new(0.08,0.9,0.2);
        train_q_learning(&mut agent,TRAIN_EPISODE);
        agent.train = false;
        (
            TicTacToeApp {
                board: Board::new(),
                game_over: false,
                winner: None,
                ai_agent: agent,
                game_mode: GameMode::PvP,
                ai_thinking: false,
                ai_turn_start: None,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Tic Tac Toe Game")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::CellClicked(row, col) => {
                if !self.game_over && !self.ai_thinking {
                    let (move_status, winner) = self.board.make_move(row, col);
                    if move_status.move_successful {
                        self.game_over = move_status.game_over;
                        self.winner = winner;

                        if !self.game_over
                            && self.game_mode == GameMode::PvA
                            && self.board.get_current_player().marker == Cell::O
                        {
                            self.ai_thinking = true;
                            self.ai_turn_start = Some(Instant::now());
                            return Command::perform(
                                async {},
                                |_| Message::AIMove,
                            )
                        }
                    }
                }
            }
            Message::ResetGame => {
                self.board.reset();
                self.game_over = false;
                self.winner = None;
                self.ai_thinking = false;
                self.ai_turn_start = None;

                if self.game_mode == GameMode::PvA
                    && self.board.get_current_player().marker == Cell::O {
                    self.ai_thinking = true;
                    self.ai_turn_start = Some(Instant::now());
                    return Command::perform(
                        async {},
                        |_| Message::AIMove,
                    )
                }
            }
            Message::AIMove => {
                let available_moves = self.board.available_moves();
                let blocking_move = self.board.find_blocking_move();
                if !available_moves.is_empty() {
                    let state = self.board.board_state();
                    let (action,_) = self.ai_agent.choose_action(&state, &available_moves,blocking_move);
                    let (move_status, winner) = self.board.make_move(action.0,action.1);
                    self.game_over = move_status.game_over;
                    self.winner = winner;
                }
                self.ai_thinking = false;
            }
            Message::Tick => {
                if self.ai_thinking {
                    if let Some(start_time) = self.ai_turn_start {
                        if start_time.elapsed() >= Duration::from_millis(500) {
                            return Command::perform(
                                async {},
                                |_| Message::AIMove,
                            )
                        }
                    }
                }
            }
            Message::SetGameMode(mode) => {
                self.game_mode = mode;
                self.board.reset();
                self.game_over = false;
                self.winner = None;
                self.ai_thinking = false;
                self.ai_turn_start = None;

                if self.game_mode == GameMode::PvA
                    && self.board.get_current_player().marker == Cell::O {
                    self.ai_thinking = true;
                    self.ai_turn_start = Some(Instant::now());
                }
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let title = text("Tic-Tac-Toe")
            .size(40)
            .width(Length::Fill)
            .horizontal_alignment(alignment::Horizontal::Center);

        // Game mode selection
        let game_mode_row = row![
            button(text("Player vs Player").horizontal_alignment(alignment::Horizontal::Center))
                .on_press(Message::SetGameMode(GameMode::PvP))
                .width(Length::Fill)
                .style(if self.game_mode == GameMode::PvP {
                    iced::theme::Button::Primary
                } else {
                    iced::theme::Button::Secondary
                }),
            button(text("Player vs AI").horizontal_alignment(alignment::Horizontal::Center))
                .on_press(Message::SetGameMode(GameMode::PvA))
                .width(Length::Fill)
                .style(if self.game_mode == GameMode::PvA {
                    iced::theme::Button::Primary
                } else {
                    iced::theme::Button::Secondary
                })
        ]
            .spacing(20);

        // Current player or game result display
        let status_text = if self.game_over {
            match self.winner {
                Some(Cell::X) => "Player X wins!",
                Some(Cell::O) => match self.game_mode {
                    GameMode::PvP => "Player O wins!",
                    GameMode::PvA => "AI wins!",
                },
                _ => "It's a draw!",
            }
        } else {
            match self.board.get_current_player().marker {
                Cell::X => "Player X's turn",
                Cell::O => match self.game_mode {
                    GameMode::PvP => "Player O's turn",
                    GameMode::PvA => {
                        if self.ai_thinking {
                            "AI is thinking..."
                        } else {
                            "AI's turn"
                        }
                    },
                },
                _ => "",
            }
        };

        let status = text(status_text)
            .size(24)
            .width(Length::Fill)
            .horizontal_alignment(alignment::Horizontal::Center);

        // Build the game grid
        let mut grid = Column::new().spacing(5).width(Length::Fill);

        for i in 0..3 {
            let mut row_widgets = row!().spacing(5).width(Length::Fill);

            for j in 0..3 {
                let cell_text = match self.board.grid[i][j] {
                    Cell::X => "X",
                    Cell::O => "O",
                    Cell::Empty => " ",
                };

                let cell_button = button(
                    text(cell_text)
                        .size(40)
                        .horizontal_alignment(alignment::Horizontal::Center)
                        .vertical_alignment(alignment::Vertical::Center),
                )
                    .width(Length::Fill)
                    .height(Length::Fixed(80.0))
                    .style(match self.board.grid[i][j] {
                        Cell::X => iced::theme::Button::Positive,
                        Cell::O => iced::theme::Button::Destructive,
                        Cell::Empty => iced::theme::Button::Secondary,
                    });

                let cell = if self.board.grid[i][j] == Cell::Empty && !self.game_over && !self.ai_thinking {
                    cell_button.on_press(Message::CellClicked(i, j))
                } else {
                    cell_button
                };

                row_widgets = row_widgets.push(cell);
            }

            grid = grid.push(row_widgets);
        }

        // Reset button
        let reset_button = button(text("New Game"))
            .on_press(Message::ResetGame)
            .width(Length::Fixed(120.0))
            .padding(10)
            .style(iced::theme::Button::Primary);

        // Main column with all components
        let content = Column::new()
            .push(title)
            .push(game_mode_row)
            .push(status)
            .push(grid)
            .push(Row::new().push(reset_button).width(Length::Fill).padding(10).align_items(alignment::Alignment::Center))
            .padding(20)
            .spacing(20)
            .width(Length::Fill)
            .max_width(500.0);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .center_y()
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        time::every(Duration::from_millis(100)).map(|_| Message::Tick)
    }
}



fn main() {
    let settings = Settings {
        antialiasing: true,
        window: iced::window::Settings {
            size: iced::Size::new(400.0,600.0),
            resizable: false,
            decorations: true,
            ..Default::default()
        },
        ..Settings::default()
    };
    TicTacToeApp::run(settings).unwrap();
}
