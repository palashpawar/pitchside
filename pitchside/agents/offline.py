"""Template writer used when no model is configured, and as the grounded fallback
when a model's output fails verification twice. Every value comes from facts."""
from __future__ import annotations


class _Facts(dict):
    def __missing__(self, key):
        return "?"


def _fmt(template: str, facts: dict) -> str:
    return template.format_map(_Facts(facts)).strip()


# kind -> (headline, why) in English: the canonical analyst insight.
ANALYST_EN = {
    "goal": ("{scorer} scores! {score}",
             "{game_state_cap} for {team}. The move began with {move_started} and needed {move_passes} passes in "
             "{move_duration_s} seconds; the finish was a {xg} xG chance.{assist_clause}"),
    "big_chance": ("Big chance for {shooter}",
                   "{shooter} had a {xg} xG opportunity from {distance_m} m ({outcome}). {team} have built "
                   "{team_xg} xG from {team_shots} shots."),
    "big_save": ("{goalkeeper} denies {shooter}",
                 "A {xg} xG chance from {distance_m} m kept out; {team} have {team_xg} xG from {team_shots} "
                 "shots and still need the breakthrough."),
    "press_shift": ("{team} crank up the press",
                    "Pressures rose from {pressures_per_min_before} to {pressures_per_min_now} per minute and PPDA "
                    "fell from {ppda_before} to {ppda_now}: they are winning the ball back sooner, {press_height_m_now} m "
                    "from their own goal."),
    "tempo_shift": ("{team} {tempo_verb} the tempo",
                    "{team} now average {passes_per_possession_min_now} passes per minute on the ball, against "
                    "{passes_per_possession_min_before} before, with {directness_pct_now}% of pass distance going forward."),
    "chaos_spell": ("The game has broken open",
                    "Turnovers rose from {turnovers_per_min_before} to {turnovers_per_min_now} per minute and "
                    "possessions last just {passes_per_possession_now} passes: neither side is in control."),
    "control_regained": ("{team} take control",
                         "{possession_share_pct}% of the ball over the last {window_min} minutes at {pass_accuracy_pct}% "
                         "passing accuracy; the chaos index fell from {chaos_index_before} to {chaos_index_now}."),
    "momentum_swing": ("Momentum swings to {team}",
                       "{final_third_entries_last_5_min} final-third entries and {xg_last_5_min} xG in {window_min} "
                       "minutes have flipped the pressure."),
    "line_breaking_pass": ("{passer} cuts through the lines",
                           "A {pass_distance_m} m pass to {receiver} took {opponents_bypassed} opponents out of the "
                           "game; the model gave it a {completion_probability_pct}% chance of arriving."),
    "speed_record": ("{player} clocks {top_speed_kmh} km/h", "The fastest sprint of the match so far."),
    "milestone": ("{player}: {passes_completed} passes completed",
                  "{pass_accuracy_pct}% accuracy from {passes_attempted} attempts: the metronome of this side."),
    "long_spell": ("{team} keep the ball",
                   "{passes_in_spell} passes over {spell_duration_s} seconds, with {players_involved} players involved."),
}

CASUAL_EN = {
    "goal": "GOAL! {scorer} makes it {score}.",
    "big_chance": "So close! {shooter} should have scored there.",
    "big_save": "What a save from {goalkeeper}!",
    "press_shift": "{team} are hunting the ball high up the pitch now.",
    "tempo_shift": "{team} have changed gear.",
    "chaos_spell": "End to end! Nobody can keep the ball right now.",
    "control_regained": "{team} have calmed things down and are running the game.",
    "momentum_swing": "{team} are on top now.",
    "line_breaking_pass": "Brilliant pass from {passer} to {receiver}!",
    "speed_record": "{player} is flying: {top_speed_kmh} km/h!",
    "milestone": "{player} hardly gives the ball away today.",
    "long_spell": "{team} are passing it around with ease.",
}

ANALYST_ES = {
    "goal": ("¡Gol de {scorer}! {score}",
             "{game_state_es} para {team}. La jugada nació de {move_started_es} y necesitó {move_passes} pases en "
             "{move_duration_s} segundos; la ocasión valía {xg} xG.{assist_clause_es}"),
    "big_chance": ("Ocasión clarísima de {shooter}",
                   "{shooter} tuvo una ocasión de {xg} xG desde {distance_m} m. {team} suma {team_xg} xG en "
                   "{team_shots} disparos."),
    "big_save": ("{goalkeeper} lo evita",
                 "Parada ante una ocasión de {xg} xG desde {distance_m} m."),
    "press_shift": ("{team} adelanta la presión",
                    "Las presiones pasan de {pressures_per_min_before} a {pressures_per_min_now} por minuto y el PPDA "
                    "baja de {ppda_before} a {ppda_now}: recuperan antes y a {press_height_m_now} m de su portería."),
    "tempo_shift": ("{team} cambia el ritmo",
                    "Ahora {passes_per_possession_min_now} pases por minuto con balón, frente a "
                    "{passes_per_possession_min_before} antes."),
    "chaos_spell": ("El partido se ha roto",
                    "Las pérdidas suben de {turnovers_per_min_before} a {turnovers_per_min_now} por minuto y las "
                    "posesiones duran {passes_per_possession_now} pases: nadie manda."),
    "control_regained": ("{team} toma el control",
                         "{possession_share_pct}% de posesión en los últimos {window_min} minutos con "
                         "{pass_accuracy_pct}% de acierto en el pase."),
    "momentum_swing": ("El impulso cambia hacia {team}",
                       "{final_third_entries_last_5_min} llegadas al último tercio y {xg_last_5_min} xG en "
                       "{window_min} minutos."),
    "line_breaking_pass": ("{passer} rompe líneas",
                           "Pase de {pass_distance_m} m a {receiver} que dejó atrás a {opponents_bypassed} rivales; "
                           "el modelo le daba un {completion_probability_pct}% de probabilidad de llegar."),
    "speed_record": ("{player} alcanza {top_speed_kmh} km/h", "El sprint más rápido del partido."),
    "milestone": ("{player}: {passes_completed} pases completados",
                  "{pass_accuracy_pct}% de acierto en {passes_attempted} intentos."),
    "long_spell": ("{team} guarda el balón",
                   "{passes_in_spell} pases durante {spell_duration_s} segundos."),
}

CASUAL_ES = {
    "goal": "¡GOOOL! {scorer} pone el {score}.",
    "big_chance": "¡Uf! {shooter} la tuvo clarísima.",
    "big_save": "¡Paradón de {goalkeeper}!",
    "press_shift": "{team} sale a por el balón arriba.",
    "tempo_shift": "{team} ha cambiado de marcha.",
    "chaos_spell": "¡Ida y vuelta! Nadie retiene el balón.",
    "control_regained": "{team} ha calmado el partido y lo controla.",
    "momentum_swing": "{team} aprieta ahora.",
    "line_breaking_pass": "¡Qué pase de {passer} para {receiver}!",
    "speed_record": "¡{player} vuela: {top_speed_kmh} km/h!",
    "milestone": "{player} no pierde un balón hoy.",
    "long_spell": "{team} mueve el balón a placer.",
}

TABLES = {"en": (ANALYST_EN, CASUAL_EN), "es": (ANALYST_ES, CASUAL_ES)}
GAME_STATE_ES = {"opening goal": "Primer gol", "equaliser": "Empate", "go-ahead goal": "Gol que pone por delante",
                 "pulls one back": "Recorta distancias", "extends the lead": "Amplía la ventaja"}


def enrich(kind: str, facts: dict) -> dict:
    f = dict(facts)
    f.setdefault("team", f.get("home_team"))
    if kind == "goal":
        f["game_state_cap"] = str(f.get("game_state", "goal")).capitalize()
        f["assist_clause"] = f" {f['assist']} provided the assist." if f.get("assist") else ""
        f["game_state_es"] = GAME_STATE_ES.get(f.get("game_state"), "Gol")
        f["assist_clause_es"] = f" Asistencia de {f['assist']}." if f.get("assist") else ""
        f["move_started_es"] = "la recuperación"
    if kind == "tempo_shift":
        f["tempo_verb"] = "raise" if f.get("direction") == "up" else "drop"
    return f


def analyst(kind: str, facts: dict, lang: str = "en") -> dict:
    table = TABLES.get(lang, TABLES["en"])[0]
    head, why = table.get(kind, ("{team}", ""))
    f = enrich(kind, facts)
    return {"headline": _fmt(head, f), "why": _fmt(why, f)}


def persona(kind: str, facts: dict, insight: dict, profile: dict, involved: bool, favours: bool | None) -> dict:
    lang = profile.get("language", "en") if profile.get("language") in TABLES else "en"
    analyst_t, casual_t = TABLES[lang]
    f = enrich(kind, facts)
    base = analyst(kind, facts, lang)
    if profile.get("mode") == "analyst":
        overlay, line = base["headline"], base["why"]
    else:
        overlay, line = base["headline"], _fmt(casual_t.get(kind, base["headline"]), f)
    metric = profile.get("metric")
    key = {"xg": "xg", "speed": "top_speed_kmh", "passing": "pass_accuracy_pct", "pressing": "pressures_per_min_now"}.get(metric or "")
    if key and key in facts:
        label = {"xg": "xG", "top_speed_kmh": "km/h", "pass_accuracy_pct": "%", "pressures_per_min_now": "press/min"}[key]
        overlay = f"{facts[key]} {label} · {overlay}"
    if favours is True:
        line = ("¡Vamos! " if lang == "es" else "Come on! ") + line
    elif favours is False and kind in ("goal", "big_chance", "momentum_swing", "press_shift"):
        line = ("Cuidado: " if lang == "es" else "Danger: ") + line
    if involved:
        overlay = "★ " + overlay
    return {"overlay": overlay, "line": line}
