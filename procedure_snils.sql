create or replace procedure procedure_task_chelindbank(snils_number INT, control_number INT)
LANGUAGE plpgsql as $$
declare 
	sum_number int := 0;
	digit TEXT;
	n_position int := LENGTH(snils_number :: TEXT); 
begin
	-- Проверка длины номера и контрольного числа 
	IF LENGTH(snils_number :: TEXT) <> 9 OR
	LENGTH(control_number :: TEXT) <> 2
	THEN
        RAISE EXCEPTION 'Ошибка: некорретный ввод СНИЛС';
    END IF;
	
	-- Бежим по числам номера СНИЛС
	FOR i IN 1..LENGTH(snils_number::TEXT) LOOP
		digit := substring(snils_number::TEXT, i, 1);
		sum_number = sum_number + digit :: INT * n_position;
		n_position = n_position - 1;
	END LOOP;

	IF sum_number <> control_number 
	THEN
		RAISE INFO 'Ошибка: Контрольное число неверно';
	ELSE
		RAISE INFO 'Успешно: Контрольное число верно';
	END IF;
end $$ 
