SELECT T.team_long_name as team_name, 
    AVG(TA.chanceCreationPassing) as pass, 
    AVG(TA.chanceCreationCrossing) as cross, 
    AVG(TA.chanceCreationShooting) as shoot, 
    AVG(TA.defencePressure) as pressure,
    AVG(TA.defenceAggression) as aggression
FROM Team_Attributes as TA
LEFT JOIN Team AS T on T.team_api_id = TA.team_api_id
WHERE T.team_api_id IN (SELECT DISTINCT home_team_api_id
                            FROM Match
                            JOIN Country on Country.id = Match.country_id
                            WHERE Country.name = 'England')
GROUP BY T.team_long_name
;