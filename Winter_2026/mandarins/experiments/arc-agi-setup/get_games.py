import arc_agi

arc = arc_agi.Arcade()
games = arc.get_environments()

for game in games:
    print(f"{game.game_id}: {game.title}")