package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

class Guest {
    private String name;

    public Guest(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}

class Room {
    private int roomNumber;
    private boolean isOccupied;
    private Guest guest;

    public Room(int roomNumber) {
        this.roomNumber = roomNumber;
        this.isOccupied = false;
        this.guest = null;
    }

    public int getRoomNumber() {
        return roomNumber;
    }

    public boolean isOccupied() {
        return isOccupied;
    }

    public Guest getGuest() {
        return guest;
    }

    public void occupy(Guest guest) {
        this.guest = guest;
        isOccupied = true;
    }

    public void vacate() {
        guest = null;
        isOccupied = false;
    }
}

class Hotel {
    private List<Room> rooms;

    public Hotel(int numberOfRooms) {
        rooms = new ArrayList<>();
        for (int i = 1; i <= numberOfRooms; i++) {
            rooms.add(new Room(i));
        }
    }

    public boolean bookRoom(int roomNumber, Guest guest) {
        for (Room room : rooms) {
            if (room.getRoomNumber() == roomNumber && !room.isOccupied()) {
                room.occupy(guest);
                return true; // Успешно
            }
        }
        return false;
    }

    public boolean vacateRoom(int roomNumber) {
        for (Room room : rooms) {
            if (room.getRoomNumber() == roomNumber && room.isOccupied()) {
                room.vacate();
                return true; // Номер освобожден
            }
        }
        return false;
    }

    public boolean hasAccess(int roomNumber, String guestName){
        for (Room room : rooms) {
            if (room.getRoomNumber() == roomNumber && room.isOccupied() && room.getGuest().getName().equals(guestName)) {
                return true;
            }
        }
        return false;
    }

    public void displayAvailableRooms() {
        System.out.println("Доступные номера:");
        for (Room room : rooms) {
            if (!room.isOccupied()) {
                System.out.println("Номер " + room.getRoomNumber());
            }
        }
    }
}

/**
 * Бронирование номеров отелей.
 *
 * Есть простая проверка по имени для освобождения номера,
 * то есть освободить может только тот, кто забронировал.
 *
 * Можно оптимизировать программу можно в поисках номеров,
 * я использовал перебор, но можно конечно использовать индексы или же хэштаблицу <String, Room>,
 * если названия номеров сложные, как например 1S для бизнес класса, 1E для эконом и так далее.
 * */
public class Main {
    public static void main(String[] args) {
        Hotel hotel = new Hotel(10); // Создаем гостиницу с 10 номерами

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("\nВыберите действие:");
            System.out.println("1. Забронировать номер");
            System.out.println("2. Освободить номер");
            System.out.println("3. Отобразить доступные номера");
            System.out.println("4. Выйти");

            int choice = scanner.nextInt();
            scanner.nextLine();

            String guestName;

            switch (choice) {
                case 1:
                    System.out.print("Введите ваше имя: ");
                    guestName = scanner.nextLine();
                    System.out.print("Введите номер, который хотите забронировать: ");
                    int roomToBook = scanner.nextInt();
                    scanner.nextLine();
                    Guest guest = new Guest(guestName);
                    if (hotel.bookRoom(roomToBook, guest)) {
                        System.out.println("Номер " + roomToBook + " забронирован для " + guest.getName());
                    } else {
                        System.out.println("Номер " + roomToBook + " не удалось забронировать. Номер уже забронирован.");
                    }
                    break;

                case 2:
                    System.out.print("Введите ваше имя: ");
                    guestName = scanner.nextLine();
                    System.out.print("Введите номер, который хотите освободить: ");
                    int roomToVacate = scanner.nextInt();
                    scanner.nextLine();
                    if(!hotel.hasAccess(roomToVacate, guestName)){
                        System.out.println("Вы не можете освободить этот номер");
                        break;
                    }
                    if (hotel.vacateRoom(roomToVacate)) {
                        System.out.println("Номер " + roomToVacate + " освобожден.");
                    } else {
                        System.out.println("Номер " + roomToVacate + " не был забронирован или уже освобожден.");
                    }
                    break;

                case 3:
                    hotel.displayAvailableRooms();
                    break;

                case 4:
                    System.out.println("Выход из программы.");
                    scanner.close();
                    System.exit(0);

                default:
                    System.out.println("Некорректный выбор.");
            }
        }
    }
}